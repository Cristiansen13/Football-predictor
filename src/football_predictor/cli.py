"""Command-line interface.

    ingest         ETL from Football-Data, Understat, FBref and SOFIFA into SQLite
    check-sources  fetch one live page per source and report what parses
    status         data coverage per season, quality issues, point-in-time audit
    ratings        fit the model and print team strengths
    predict        probabilities (and EV against current odds) for fixtures
    backtest       walk-forward test season: scoring rules, calibration, EV betting
    demo           backtest on a simulated league (no network needed)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings, season_from_date
from .db import Database
from .etl.browser import BrowserFetcher
from .etl.entity_resolution import team_resolver
from .etl.http import HttpClient
from .etl.pipeline import DIVISIONS, SOURCES, Pipeline, probe_sources
from .evaluation import (
    BettingStrategy,
    expected_value,
    kelly_fraction,
    run_backtest,
    save_report,
    tune_hyperparameters,
)
from .features import audit_point_in_time, build_features, lineup_strength
from .models import FEATURE_PRESETS, DixonColesModel
from .quality import coverage_report, validate
from .synthetic import simulate_league

TUNING_GRID = {"xi": [0.001, 0.0015, 0.002, 0.003, 0.004], "xg_weight": [0.0, 0.3, 0.6]}
FIRST_SEASON = 2014


def current_season() -> int:
    return season_from_date(date.today())


def parse_seasons(text: str) -> list[int]:
    """'2018-2023' or '2019,2021' -> list of season start years."""
    seasons: list[int] = []
    for part in text.split(","):
        if "-" in part:
            first, last = (int(x) for x in part.split("-"))
            seasons.extend(range(first, last + 1))
        else:
            seasons.append(int(part))
    return seasons


def _csv_list(text: str | None) -> tuple[str, ...] | None:
    return tuple(t.strip() for t in text.split(",") if t.strip()) if text else None


# -- data -------------------------------------------------------------------------


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """(matches, all-competition schedule, lineup strength) from the warehouse or a simulation."""
    if args.synthetic:
        return simulate_league(seed=args.seed, second_division_teams=20), None, None
    db = Database(args.db)
    matches = db.load_match_dataset()
    if matches.empty:
        raise SystemExit(f"No matches in {args.db}; run `football-predictor ingest` first or pass --synthetic")
    names = dict(zip(matches["home_team_id"], matches["home_team"])) | dict(
        zip(matches["away_team_id"], matches["away_team"])
    )
    schedule = db.load_team_schedule()
    lineups = lineup_strength(db.load_lineups(), db.load_player_ratings())
    lineups = lineups.assign(team=lineups["team_id"].map(names))
    db.close()
    if args.train_divisions:
        keep = set(args.train_divisions) | {args.division}
        matches = matches[matches["division"].isin(keep)]
    return matches, schedule, lineups


def featurize(args: argparse.Namespace, matches, schedule=None, lineups=None) -> pd.DataFrame:
    return build_features(matches, schedule=schedule, lineups=lineups, halflife_days=args.halflife)


def load_features(args: argparse.Namespace) -> pd.DataFrame:
    """Point-in-time feature table for every fixture."""
    return featurize(args, *load_inputs(args))


def model_kwargs(args: argparse.Namespace) -> dict:
    attack, defence = FEATURE_PRESETS[args.features]
    return {
        "xi": args.xi,
        "xg_weight": args.xg_weight,
        "l2": args.l2,
        "attack_features": _csv_list(args.attack_features) or attack,
        "defence_features": _csv_list(args.defence_features) or defence,
        "default_division": args.division,
    }


def model_factory(args: argparse.Namespace):
    return partial(DixonColesModel, **model_kwargs(args))


def make_client(args: argparse.Namespace) -> tuple[HttpClient, BrowserFetcher | None]:
    settings = Settings(data_dir=args.data_dir)
    browser = None
    if not args.no_browser_fallback:
        browser = BrowserFetcher(executable_path=args.chromium_path, headless=not args.headful)
    return HttpClient(settings.cache_dir, settings.request_delay, fallback=browser), browser


def browser_kwargs(args: argparse.Namespace) -> dict:
    return {"executable_path": args.chromium_path, "headless": not args.headful}


# -- commands ---------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> None:
    db = Database(args.db)
    client, browser = make_client(args)
    sources = SOURCES if "all" in args.sources else args.sources
    pipeline = Pipeline(db, client, current_season=current_season(), divisions=args.divisions)
    try:
        failures = pipeline.run(
            parse_seasons(args.seasons), sources, lineups=args.lineups, upcoming=not args.no_upcoming,
            browser_kwargs=browser_kwargs(args), sofifa_every_days=args.sofifa_every_days,
        )
    finally:
        if browser is not None:
            browser.close()
    for source, season, error in failures:
        print(f"FAILED {source} {season or ''}: {error}")
    report = coverage_report(db)
    db.close()
    print_coverage(report)
    if failures:
        raise SystemExit(1)


def cmd_check_sources(args: argparse.Namespace) -> None:
    client, browser = make_client(args)
    season = args.season if args.season is not None else current_season() - 1
    sources = SOURCES if "all" in args.sources else args.sources
    try:
        results = probe_sources(client, season, sources, browser_kwargs=browser_kwargs(args))
    finally:
        if browser is not None:
            browser.close()
    print(f"Live check for season {season}/{(season + 1) % 100:02d}:")
    for row in results.itertuples(index=False):
        print(f"  {'OK  ' if row.ok else 'FAIL'} {row.source:<22} rows={row.rows:<5} {row.detail}")
    if not results["ok"].all():
        raise SystemExit(1)


def print_coverage(report: pd.DataFrame) -> None:
    if report.empty:
        print("Database is empty.")
        return
    shown = report.copy()
    for column in ("xg", "ppda", "possession", "shots", "closing_odds", "any_odds", "lineups"):
        shown[column] = (shown[column] * 100).round().astype(int).astype(str) + "%"
    print(shown.to_string(index=False))


def cmd_status(args: argparse.Namespace) -> None:
    db = Database(args.db)
    report = coverage_report(db)
    db.close()
    print_coverage(report)
    issues = validate(report, current_season())
    print("\nNo data issues found." if not issues else "\n" + "\n".join(map(str, issues)))
    if args.audit:
        args.synthetic, args.train_divisions = False, None
        matches, schedule, lineups = load_inputs(args)
        problems = audit_point_in_time(
            matches, schedule=schedule, lineups=lineups, sample=args.audit, halflife_days=args.halflife
        )
        if problems.empty:
            print(f"\nPoint-in-time audit: {args.audit} random matches, every feature reproduced "
                  "from pre-kickoff data only.")
        else:
            print(f"\nPoint-in-time audit FAILED: {len(problems)} feature values depend on the future")
            print(problems.head(20).to_string(index=False))
            raise SystemExit(1)


def _division_teams(model: DixonColesModel) -> set[str]:
    return set(model.teams_[model.reference_index_])


def cmd_ratings(args: argparse.Namespace) -> None:
    data = load_features(args)
    model = model_factory(args)().fit(data, as_of=args.as_of)
    s = model.summary_
    print(f"Fitted on {s.n_matches} matches ({', '.join(s.divisions)}) before {s.as_of:%Y-%m-%d} "
          f"(effective {s.effective_matches:.0f}); {model.default_division_} home advantage "
          f"{model.home_advantage_:+.3f}, rho {model.rho_:+.3f}, xg_weight {model.xg_weight}")
    if model.theta_attack_ or model.theta_defence_:
        print("feature coefficients:", json.dumps(
            {"attack": {k: round(float(v), 4) for k, v in model.theta_attack_.items()},
             "defence": {k: round(float(v), 4) for k, v in model.theta_defence_.items()}}))
    ratings = model.ratings()
    if not args.all_teams:
        ratings = ratings[ratings["team"].isin(_division_teams(model))]
    print(ratings.round(3).to_string(index=False))


def _fixture_rows(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(feature table, fixtures to predict)."""
    matches, schedule, lineups = load_inputs(args)
    if args.upcoming is not None:
        data = featurize(args, matches, schedule, lineups)
        now = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now()
        upcoming = data[data["home_goals"].isna() & (data["kickoff"] >= now)
                        & (data["kickoff"] < now + pd.Timedelta(days=args.upcoming))]
        if "division" in upcoming:
            upcoming = upcoming[upcoming["division"] == args.division]
        if upcoming.empty:
            raise SystemExit(f"No unplayed {args.division} fixtures in the next {args.upcoming} days "
                             "(run `football-predictor ingest` to refresh fixtures)")
        return data, upcoming
    if not (args.home and args.away):
        raise SystemExit("Pass HOME and AWAY teams, or --upcoming DAYS")

    known = set(matches["home_team"]) | set(matches["away_team"])
    resolver = team_resolver()
    home, away = (
        name if name in known else resolver.resolve(name, create_missing=True).canonical
        for name in (args.home, args.away)
    )
    scheduled = matches[(matches["home_team"] == home) & (matches["away_team"] == away)
                        & matches["home_goals"].isna()]
    if scheduled.empty:
        # Not in the database: add it so its features are built like any other fixture.
        kickoff = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now().floor("min")
        query = pd.DataFrame([{"match_id": "query", "season": season_from_date(kickoff), "division": args.division,
                               "kickoff": kickoff, "home_team": home, "away_team": away,
                               "home_goals": pd.NA, "away_goals": pd.NA}])
        matches = pd.concat([matches, query], ignore_index=True)
        target = "query"
    else:
        target = scheduled["match_id"].iloc[0]
    data = featurize(args, matches, schedule, lineups)
    return data, data[data["match_id"] == target]


def _with_value(fixtures: pd.DataFrame, predictions: pd.DataFrame, kelly_multiplier: float) -> pd.DataFrame:
    """Best-EV selection against the stored prices, where the fixture has them."""
    odds_columns = ["odds_home", "odds_draw", "odds_away"]
    if not all(c in fixtures for c in odds_columns) or fixtures[odds_columns].isna().all().all():
        return predictions
    odds = fixtures[odds_columns].to_numpy(dtype=float)
    probabilities = predictions[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    ev = np.where(np.isnan(odds), -np.inf, expected_value(probabilities, np.nan_to_num(odds, nan=1.0)))
    pick = ev.argmax(axis=1)
    rows = np.arange(len(pick))
    out = predictions.copy()
    out[odds_columns] = odds
    out["best_bet"] = np.where(np.isfinite(ev[rows, pick]), np.array(["home", "draw", "away"])[pick], "-")
    out["best_ev"] = np.where(np.isfinite(ev[rows, pick]), ev[rows, pick], np.nan)
    stake = kelly_multiplier * kelly_fraction(probabilities[rows, pick], np.nan_to_num(odds[rows, pick], nan=1.0))
    out["stake_pct"] = np.where(out["best_ev"] > 0, 100 * stake, 0.0)
    return out


def cmd_predict(args: argparse.Namespace) -> None:
    data, fixtures = _fixture_rows(args)
    model = model_factory(args)().fit(data, as_of=fixtures["kickoff"].min())
    predictions = pd.concat([fixtures[["kickoff", "home_team", "away_team"]], model.predict(fixtures)], axis=1)
    predictions = _with_value(fixtures, predictions, args.kelly)
    predictions["kickoff"] = predictions["kickoff"].dt.strftime("%Y-%m-%d %H:%M")
    print(predictions.round(3).to_string(index=False))
    if len(fixtures) == 1:
        row = fixtures.iloc[0]
        features = {c: row[c] for c in fixtures.columns if c.startswith(("home_", "away_"))
                    and c not in ("home_team", "away_team", "home_goals", "away_goals")}
        matrix = model.score_matrix(row["home_team"], row["away_team"], division=row.get("division", args.division),
                                    **features)
        print("\nCorrect-score probabilities (%):")
        print((matrix * 100).round(1).to_string())


METRIC_SECTIONS = {
    "Forecast quality": ["matches", "brier_model", "rps_model", "log_loss_model", "score_log_loss_model",
                         "accuracy_model", "ece_model", "brier_base_rate", "brier_skill_vs_base_rate"],
    "Against the market": ["matches_with_odds", "brier_model_priced", "brier_market", "brier_skill_vs_market",
                           "brier_diff_vs_market", "brier_diff_ci_low", "brier_diff_ci_high", "rps_market",
                           "log_loss_market", "accuracy_market", "ece_market", "mean_overround"],
    "Betting": ["bets", "staked", "profit", "roi", "final_bankroll", "bankroll_growth", "max_drawdown",
                "hit_rate", "mean_odds", "mean_ev", "flat_stake_profit_units", "flat_stake_roi",
                "flat_roi_ci_low", "flat_roi_ci_high", "p_flat_roi_le_zero", "bets_home", "roi_home",
                "bets_draw", "roi_draw", "bets_away", "roi_away"],
}


def print_metrics(metrics: dict) -> None:
    for section, keys in METRIC_SECTIONS.items():
        print(f"  {section}")
        for key in keys:
            if key in metrics:
                value = metrics[key]
                text = f"{value:.4f}" if isinstance(value, float) else str(value)
                print(f"    {key:>26}: {text}")


def _season_window(data: pd.DataFrame, season: int, division: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    rows = data[(data["season"] == season) & (data["division"] == division)]
    if rows.empty:
        raise SystemExit(f"No {division} matches in season {season}")
    return rows["kickoff"].min().normalize(), rows["kickoff"].max() + pd.Timedelta(seconds=1)


def default_test_season(data: pd.DataFrame, division: str) -> int:
    """The latest season of ``division`` with every fixture played.

    During a season the warehouse holds the whole fixture list with results for
    the rounds played so far; scoring those few matches says little, so the
    previous, complete season is used instead. The latest season is kept when
    no season is complete (the first season of a fresh database).
    """
    rows = data[data["division"] == division]
    if rows.empty:
        raise SystemExit(f"No {division} matches loaded")
    unplayed = rows.groupby("season")["home_goals"].apply(lambda g: g.isna().any())
    complete = unplayed[~unplayed].index
    return int(complete.max() if len(complete) else rows["season"].max())


def cmd_backtest(args: argparse.Namespace) -> None:
    data = load_features(args)
    divisions = [args.division]
    test_season = args.test_season if args.test_season is not None else default_test_season(data, args.division)
    start, end = _season_window(data, test_season, args.division)

    if args.tune:
        val_start, _ = _season_window(data, test_season - 1, args.division)
        base = {k: v for k, v in model_kwargs(args).items() if k not in TUNING_GRID}
        scores = tune_hyperparameters(
            data[data["kickoff"] < start], TUNING_GRID, val_start, start, model_kwargs=base, divisions=divisions,
        )
        print(f"Hyperparameters validated on season {test_season - 1} (best first):")
        print(scores[list(TUNING_GRID) + ["brier_model", "log_loss_model", "score_log_loss_model"]]
              .head(8).round(5).to_string(index=False))
        args.xi = float(scores.loc[0, "xi"])
        args.xg_weight = float(scores.loc[0, "xg_weight"])

    strategy = BettingStrategy(
        ev_threshold=args.ev_threshold,
        kelly_multiplier=args.kelly,
        max_stake_fraction=args.max_stake,
        max_odds=args.max_odds,
    )
    reports = Path(args.reports_dir)
    if args.compare:
        rows = []
        for preset in FEATURE_PRESETS:
            kwargs = {**model_kwargs(args), "attack_features": FEATURE_PRESETS[preset][0],
                      "defence_features": FEATURE_PRESETS[preset][1]}
            result = run_backtest(data, partial(DixonColesModel, **kwargs), start, end, refit_days=args.refit_days,
                                  strategy=strategy, initial_bankroll=args.bankroll, divisions=divisions)
            rows.append({"features": preset, **result.metrics})
        table = pd.DataFrame(rows)
        columns = ["features", "brier_model", "brier_market", "log_loss_model", "score_log_loss_model",
                   "ece_model", "bets", "roi", "flat_stake_roi", "p_flat_roi_le_zero"]
        print(f"Feature presets on season {test_season} (xi={args.xi}, xg_weight={args.xg_weight}):")
        print(table[[c for c in columns if c in table]].round(4).to_string(index=False))
        reports.mkdir(parents=True, exist_ok=True)
        table.to_csv(reports / f"comparison_{test_season}.csv", index=False)

    result = run_backtest(
        data, model_factory(args), start, end,
        refit_days=args.refit_days, strategy=strategy, initial_bankroll=args.bankroll, divisions=divisions,
    )
    print(f"Test season {test_season} ({args.division}) | xi={args.xi} | xg_weight={args.xg_weight} | "
          f"features={args.features}")
    print_metrics(result.metrics)
    label = f"{test_season}"
    paths = save_report(result, reports, label,
                        title=f"Cumulative PnL, season {test_season}/{(test_season + 1) % 100:02d}")
    if args.plot:
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        Path(args.plot).write_bytes(paths["pnl_plot"].read_bytes())
    print(f"Reports: {reports} (metrics, predictions, bets, calibration, PnL and calibration plots)")


# -- parser -----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    settings = Settings()
    parser = argparse.ArgumentParser(
        prog="football-predictor", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--db", type=Path, default=None, help="SQLite path (default: <data-dir>/football.db)")
    sub = parser.add_subparsers(dest="command", required=True)

    def network_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--chromium-path", default=None, help="Chromium executable for Playwright")
        p.add_argument("--headful", action="store_true", help="show the browser (helps with anti-bot checks)")
        p.add_argument("--no-browser-fallback", action="store_true",
                       help="do not retry blocked pages in a headless browser")

    source_choices = list(SOURCES) + ["all"]
    ingest = sub.add_parser("ingest", help="run the ETL pipeline into SQLite")
    ingest.add_argument("--seasons", default=f"{FIRST_SEASON}-{current_season()}",
                        help="season start years, e.g. 2016-2024 or 2022,2023 (default: %(default)s)")
    ingest.add_argument("--sources", nargs="+", default=["football_data", "understat"], choices=source_choices)
    ingest.add_argument("--divisions", nargs="+", default=list(DIVISIONS),
                        help="Football-Data divisions (E0 Premier League, E1 Championship)")
    ingest.add_argument("--lineups", action="store_true", help="also scrape starting XIs (one request per match)")
    ingest.add_argument("--no-upcoming", action="store_true", help="skip this week's fixtures and prices")
    ingest.add_argument("--sofifa-every-days", type=int, default=28, help="spacing of rating snapshots")
    network_options(ingest)
    ingest.set_defaults(func=cmd_ingest)

    check = sub.add_parser("check-sources", help="fetch one live page per source and report what parses")
    check.add_argument("--season", type=int, default=None, help="season to probe (default: last completed)")
    check.add_argument("--sources", nargs="+", default=list(SOURCES), choices=source_choices)
    network_options(check)
    check.set_defaults(func=cmd_check_sources)

    status = sub.add_parser("status", help="data coverage, quality issues and point-in-time audit")
    status.add_argument("--audit", type=int, default=0, metavar="N", help="audit N random matches for leakage")
    status.add_argument("--halflife", type=float, default=45.0)
    status.add_argument("--division", default="E0")
    status.set_defaults(func=cmd_status)

    def model_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--synthetic", action="store_true", help="use a simulated league instead of the database")
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--xi", type=float, default=0.0019, help="time-decay per day")
        p.add_argument("--xg-weight", type=float, default=0.3,
                       help="share of the likelihood taken from xG (0 = classic Dixon-Coles)")
        p.add_argument("--l2", type=float, default=0.5)
        p.add_argument("--features", choices=sorted(FEATURE_PRESETS), default="none")
        p.add_argument("--attack-features", default=None, help="comma list overriding the preset")
        p.add_argument("--defence-features", default=None, help="comma list overriding the preset")
        p.add_argument("--halflife", type=float, default=45.0, help="EWMA half-life in days")
        p.add_argument("--division", default="E0", help="division to rate / predict / evaluate")
        p.add_argument("--train-divisions", nargs="+", default=None,
                       help="divisions to train on (default: all loaded)")

    ratings = sub.add_parser("ratings", help="fit the model and print team strengths")
    model_options(ratings)
    ratings.add_argument("--as-of", default=None)
    ratings.add_argument("--all-teams", action="store_true", help="include clubs outside --division")
    ratings.set_defaults(func=cmd_ratings)

    predict = sub.add_parser("predict", help="probability matrix, markets and EV for fixtures")
    model_options(predict)
    predict.add_argument("home", nargs="?")
    predict.add_argument("away", nargs="?")
    predict.add_argument("--as-of", default=None, help="kickoff / fit date (default: now)")
    predict.add_argument("--upcoming", type=int, default=None, metavar="DAYS",
                         help="predict every unplayed fixture in the next DAYS days")
    predict.add_argument("--kelly", type=float, default=0.25, help="fraction of full Kelly for stake_pct")
    predict.set_defaults(func=cmd_predict)

    for name, help_text in (("backtest", "walk-forward backtest with EV betting"),
                            ("demo", "backtest on a simulated league (no network needed)")):
        bt = sub.add_parser(name, help=help_text)
        model_options(bt)
        bt.add_argument("--test-season", type=int, default=None)
        bt.add_argument("--tune", "--tune-xi", dest="tune", action="store_true",
                        help="pick xi and xg_weight on the season before the test season")
        bt.add_argument("--compare", action="store_true", help="also compare every feature preset")
        bt.add_argument("--refit-days", type=int, default=7)
        bt.add_argument("--ev-threshold", type=float, default=0.05)
        bt.add_argument("--kelly", type=float, default=0.25, help="fraction of full Kelly")
        bt.add_argument("--max-stake", type=float, default=0.05, help="max bankroll fraction per bet")
        bt.add_argument("--max-odds", type=float, default=15.0)
        bt.add_argument("--bankroll", type=float, default=1000.0)
        bt.add_argument("--plot", default=None, help="extra copy of the PnL curve at this path")
        bt.set_defaults(func=cmd_backtest, synthetic=name == "demo")
    return parser


NETWORK_COMMANDS = {"ingest", "check-sources"}


def _utf8_console() -> None:
    """Windows consoles default to a legacy code page; team names and error
    messages carry characters it cannot encode, which would abort a report."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None and (stream.encoding or "").lower().replace("-", "") != "utf8":
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> None:
    _utf8_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in NETWORK_COMMANDS:
        level = logging.DEBUG if args.verbose else logging.INFO  # show progress of long downloads
    else:
        level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    settings = Settings(data_dir=args.data_dir)
    args.db = args.db or settings.db_path
    args.reports_dir = settings.reports_dir
    try:
        args.func(args)
    except BrokenPipeError:  # output piped into e.g. `head`
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())


if __name__ == "__main__":
    main()
