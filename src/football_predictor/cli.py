"""Command-line interface: ingest, ratings, predict, backtest, demo."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from functools import partial
from pathlib import Path

import pandas as pd

from .config import Settings, season_from_date
from .db import Database
from .etl.entity_resolution import team_resolver
from .etl.http import HttpClient
from .etl.pipeline import SOURCES, Pipeline
from .evaluation import BettingStrategy, plot_pnl, run_backtest, tune_xi
from .features import build_features, lineup_strength
from .models import FEATURE_PRESETS, DixonColesModel
from .synthetic import simulate_league

XI_GRID = [0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.006]


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


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """(matches, all-competition schedule, lineup strength) from the warehouse or a simulation."""
    if args.synthetic:
        return simulate_league(seed=args.seed), None, None
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
    return matches, schedule, lineups


def featurize(args: argparse.Namespace, matches, schedule=None, lineups=None) -> pd.DataFrame:
    return build_features(matches, schedule=schedule, lineups=lineups, halflife_days=args.halflife)


def load_features(args: argparse.Namespace) -> pd.DataFrame:
    """Point-in-time feature table for every fixture."""
    return featurize(args, *load_inputs(args))


def model_factory(args: argparse.Namespace):
    return partial(DixonColesModel.from_preset, args.features, xi=args.xi, l2=args.l2)


# -- commands -------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> None:
    settings = Settings(data_dir=args.data_dir)
    db = Database(args.db)
    client = HttpClient(settings.cache_dir, settings.request_delay)
    current = season_from_date(date.today())
    pipeline = Pipeline(db, client, current_season=current)
    browser = {"executable_path": args.chromium_path} if args.chromium_path else {}
    failures = pipeline.run(parse_seasons(args.seasons), args.sources, lineups=args.lineups, browser_kwargs=browser)
    for source, season, error in failures:
        print(f"FAILED {source} {season or ''}: {error}")
    counts = db.query(
        "SELECT (SELECT COUNT(*) FROM matches) AS matches, (SELECT COUNT(*) FROM teams) AS teams, "
        "(SELECT COUNT(*) FROM odds) AS odds, (SELECT COUNT(*) FROM advanced_stats) AS advanced_rows"
    )
    print(counts.to_string(index=False))
    db.close()


def cmd_ratings(args: argparse.Namespace) -> None:
    data = load_features(args)
    model = model_factory(args)().fit(data, as_of=args.as_of)
    print(f"Fitted on {model.summary_.n_matches} matches before {model.summary_.as_of:%Y-%m-%d} "
          f"(effective {model.summary_.effective_matches:.0f}); home advantage "
          f"{model.home_advantage_:+.3f}, rho {model.rho_:+.3f}")
    if model.theta_attack_ or model.theta_defence_:
        print("feature coefficients:", json.dumps(
            {"attack": {k: round(float(v), 4) for k, v in model.theta_attack_.items()},
             "defence": {k: round(float(v), 4) for k, v in model.theta_defence_.items()}}))
    print(model.ratings().round(3).to_string(index=False))


def _fixture_rows(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(feature table, fixtures to predict)."""
    matches, schedule, lineups = load_inputs(args)
    if args.upcoming is not None:
        data = featurize(args, matches, schedule, lineups)
        now = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now()
        upcoming = data[data["home_goals"].isna() & (data["kickoff"] >= now)
                        & (data["kickoff"] < now + pd.Timedelta(days=args.upcoming))]
        if upcoming.empty:
            raise SystemExit(f"No unplayed fixtures in the next {args.upcoming} days")
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
        query = pd.DataFrame([{"match_id": "query", "season": season_from_date(kickoff), "kickoff": kickoff,
                               "home_team": home, "away_team": away, "home_goals": pd.NA, "away_goals": pd.NA}])
        matches = pd.concat([matches, query], ignore_index=True)
        target = "query"
    else:
        target = scheduled["match_id"].iloc[0]
    data = featurize(args, matches, schedule, lineups)
    return data, data[data["match_id"] == target]


def cmd_predict(args: argparse.Namespace) -> None:
    data, fixtures = _fixture_rows(args)
    model = model_factory(args)().fit(data, as_of=fixtures["kickoff"].min())
    predictions = pd.concat([fixtures[["kickoff", "home_team", "away_team"]], model.predict(fixtures)], axis=1)
    predictions["kickoff"] = predictions["kickoff"].dt.strftime("%Y-%m-%d %H:%M")
    print(predictions.round(3).to_string(index=False))
    if len(fixtures) == 1:
        row = fixtures.iloc[0]
        features = {c: row[c] for c in fixtures.columns if c.startswith(("home_", "away_"))
                    and c not in ("home_team", "away_team", "home_goals", "away_goals")}
        matrix = model.score_matrix(row["home_team"], row["away_team"], **features)
        print("\nCorrect-score probabilities (%):")
        print((matrix * 100).round(1).to_string())


def cmd_backtest(args: argparse.Namespace) -> None:
    data = load_features(args)
    seasons = sorted(data["season"].unique())
    test_season = args.test_season or seasons[-1]
    start = data.loc[data["season"] == test_season, "kickoff"].min().normalize()
    end = data.loc[data["season"] == test_season, "kickoff"].max() + pd.Timedelta(seconds=1)

    if args.tune_xi:
        validation = data[data["season"] == test_season - 1]
        if validation.empty:
            raise SystemExit("--tune-xi needs the season before the test season")
        scores = tune_xi(
            data[data["kickoff"] < start], XI_GRID, validation["kickoff"].min().normalize(), start,
            model_kwargs={"l2": args.l2, "attack_features": FEATURE_PRESETS[args.features][0],
                          "defence_features": FEATURE_PRESETS[args.features][1]},
        )
        print("xi validation (season %d):\n%s" % (test_season - 1, scores[["xi", "brier_model", "log_loss_model"]]
                                                  .round(5).to_string(index=False)))
        args.xi = float(scores.loc[0, "xi"])

    strategy = BettingStrategy(
        ev_threshold=args.ev_threshold,
        kelly_multiplier=args.kelly,
        max_stake_fraction=args.max_stake,
        max_odds=args.max_odds,
    )
    result = run_backtest(
        data, model_factory(args), start, end,
        refit_days=args.refit_days, strategy=strategy, initial_bankroll=args.bankroll,
    )
    print(f"Test season {test_season} | xi={args.xi} | features={args.features}")
    for key, value in result.metrics.items():
        print(f"  {key:>24}: {value:.4f}" if isinstance(value, float) else f"  {key:>24}: {value}")

    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    plot_path = Path(args.plot) if args.plot else reports / f"pnl_{test_season}.png"
    plot_pnl(result, plot_path, title=f"Cumulative PnL, season {test_season}/{(test_season + 1) % 100:02d}")
    result.bets.to_csv(reports / f"bets_{test_season}.csv", index=False)
    result.predictions.to_csv(reports / f"predictions_{test_season}.csv", index=False)
    print(f"PnL curve: {plot_path}\nBets and predictions: {reports}")


# -- parser ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    settings = Settings()
    parser = argparse.ArgumentParser(prog="football-predictor", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--db", type=Path, default=None, help="SQLite path (default: <data-dir>/football.db)")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="run the ETL pipeline into SQLite")
    ingest.add_argument("--seasons", required=True, help="e.g. 2016-2024 or 2022,2023 (season start years)")
    ingest.add_argument("--sources", nargs="+", default=["football_data", "understat"], choices=SOURCES)
    ingest.add_argument("--lineups", action="store_true", help="also scrape starting XIs (one request per match)")
    ingest.add_argument("--chromium-path", default=None, help="Chromium executable for SOFIFA (Playwright)")
    ingest.set_defaults(func=cmd_ingest)

    def model_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--synthetic", action="store_true", help="use a simulated league instead of the database")
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--xi", type=float, default=0.0019, help="time-decay per day")
        p.add_argument("--l2", type=float, default=0.5)
        p.add_argument("--features", choices=sorted(FEATURE_PRESETS), default="none")
        p.add_argument("--halflife", type=float, default=45.0, help="EWMA half-life in days")

    ratings = sub.add_parser("ratings", help="fit the model and print team strengths")
    model_options(ratings)
    ratings.add_argument("--as-of", default=None)
    ratings.set_defaults(func=cmd_ratings)

    predict = sub.add_parser("predict", help="probability matrix and markets for fixtures")
    model_options(predict)
    predict.add_argument("home", nargs="?")
    predict.add_argument("away", nargs="?")
    predict.add_argument("--as-of", default=None, help="kickoff / fit date (default: now)")
    predict.add_argument("--upcoming", type=int, default=None, metavar="DAYS",
                         help="predict every unplayed fixture in the next DAYS days")
    predict.set_defaults(func=cmd_predict)

    for name, help_text in (("backtest", "walk-forward backtest with EV betting"),
                            ("demo", "backtest on a simulated league (no network needed)")):
        bt = sub.add_parser(name, help=help_text)
        model_options(bt)
        bt.add_argument("--test-season", type=int, default=None)
        bt.add_argument("--tune-xi", action="store_true", help="pick xi on the season before the test season")
        bt.add_argument("--refit-days", type=int, default=7)
        bt.add_argument("--ev-threshold", type=float, default=0.05)
        bt.add_argument("--kelly", type=float, default=0.25, help="fraction of full Kelly")
        bt.add_argument("--max-stake", type=float, default=0.05, help="max bankroll fraction per bet")
        bt.add_argument("--max-odds", type=float, default=15.0)
        bt.add_argument("--bankroll", type=float, default=1000.0)
        bt.add_argument("--plot", default=None, help="PNG path for the PnL curve")
        bt.set_defaults(func=cmd_backtest, synthetic=name == "demo")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    args.db = args.db or Settings(data_dir=args.data_dir).db_path
    args.reports_dir = Settings(data_dir=args.data_dir).reports_dir
    args.func(args)


if __name__ == "__main__":
    main()
