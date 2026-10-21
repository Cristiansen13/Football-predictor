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
from .evaluation.backtest import BacktestResult, evaluate_forecasts
from .features import audit_point_in_time, build_features, lineup_strength, player_form_strength
from .models import FEATURE_PRESETS, DixonColesModel
from .quality import coverage_report, validate
from .synthetic import simulate_league

TUNING_GRID = {"xi": [0.002, 0.003, 0.004, 0.006], "xg_weight": [0.3, 0.6, 0.8]}
FIRST_SEASON = 2014
DEFAULT_JOBS = max(1, (os.cpu_count() or 2) - 2)


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
    lineups = squad_strength(db.load_lineups(), db.load_player_ratings())
    lineups = lineups.assign(team=lineups["team_id"].map(names))
    db.close()
    train = args.train_divisions or [args.division]
    if "all" not in train:
        matches = matches[matches["division"].isin(set(train) | {args.division})]
    return matches, schedule, lineups


def squad_strength(lineups: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Starting-XI strength from SOFIFA ratings when loaded, otherwise from the
    starters' own Understat xG+xA form (ratings are behind an anti-bot wall)."""
    if not ratings.empty:
        return lineup_strength(lineups, ratings)
    return player_form_strength(lineups)


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
        args.synthetic, args.train_divisions = False, ["all"]
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


def _score_grid(lam: float, mu: float, model: DixonColesModel, home: str, away: str) -> pd.DataFrame:
    from .models import score_matrices

    goals = range(model.max_goals + 1)
    return pd.DataFrame(score_matrices(lam, mu, model.rho_, model.max_goals)[0],
                        index=pd.Index(goals, name=f"{home} goals"), columns=pd.Index(goals, name=f"{away} goals"))


def cmd_predict(args: argparse.Namespace) -> None:
    data, fixtures = _fixture_rows(args)
    model = model_factory(args)().fit(data, as_of=fixtures["kickoff"].min())
    predictions = pd.concat([fixtures[["kickoff", "home_team", "away_team"]], model.predict(fixtures)], axis=1)
    predictions = _with_value(fixtures, predictions, args.kelly)
    predictions["kickoff"] = predictions["kickoff"].dt.strftime("%Y-%m-%d %H:%M")
    anchored = model.predict_anchored(fixtures)
    print("Model forecast:")
    print(predictions.round(3).to_string(index=False))
    print("\nMarket-anchored forecast (rates that reproduce the current prices; 'model' where a fixture has none):")
    shown = pd.concat([predictions[["kickoff", "home_team", "away_team"]], anchored], axis=1)
    print(shown.round(3).to_string(index=False))
    if len(fixtures) == 1:
        row = fixtures.iloc[0]
        features = {c: row[c] for c in fixtures.columns if c.startswith(("home_", "away_"))
                    and c not in ("home_team", "away_team", "home_goals", "away_goals")}
        matrix = model.score_matrix(row["home_team"], row["away_team"], division=row.get("division", args.division),
                                    **features)
        print("\nCorrect-score probabilities (%), model:")
        print((matrix * 100).round(1).to_string())
        if anchored["source"].iloc[0] != "model":
            grid = _score_grid(float(anchored["exp_home_goals"].iloc[0]), float(anchored["exp_away_goals"].iloc[0]),
                               model, row["home_team"], row["away_team"])
            print("\nCorrect-score probabilities (%), market-anchored:")
            print((grid * 100).round(1).to_string())


def cmd_scan(args: argparse.Namespace) -> None:
    """Value board: every bookmaker's price on the coming fixtures against a fair value."""
    from .evaluation.metrics import implied_probabilities
    from .models import score_matrices
    from .models.probability import handicap_effective_probabilities

    data, fixtures = _fixture_rows(args)
    model = model_factory(args)().fit(data, as_of=fixtures["kickoff"].min())
    forecast = model.predict_anchored(fixtures) if args.fair == "anchored" else model.predict(fixtures)
    lam, mu = forecast["exp_home_goals"].to_numpy(dtype=float), forecast["exp_away_goals"].to_numpy(dtype=float)
    grids = score_matrices(lam, mu, model.rho_, model.max_goals)
    db = Database(args.db)
    prices = db.load_prices(fixtures["match_id"])
    db.close()
    teams = fixtures.set_index("match_id")[["kickoff", "home_team", "away_team"]]
    index = {m: i for i, m in enumerate(fixtures["match_id"])}
    rows = []

    def add(match_id, market, selection, line, bookmaker, when, odds, fair):
        if not np.isfinite(odds) or odds <= 1.0 or not np.isfinite(fair):
            return
        ev = fair * odds - 1.0
        rows.append({"kickoff": teams.loc[match_id, "kickoff"], "match": f"{teams.loc[match_id, 'home_team']} v "
                     f"{teams.loc[match_id, 'away_team']}", "market": market, "selection": selection, "line": line,
                     "bookmaker": bookmaker, "prices": when, "odds": odds, "fair_prob": fair, "ev": ev,
                     "stake_pct": 100 * args.kelly * max(kelly_fraction(fair, odds), 0.0)})

    for row in prices["1x2"].itertuples(index=False):
        i = index.get(row.match_id)
        if i is None or row.bookmaker == "max" and not args.include_max:
            continue
        for selection, odds in (("home", row.home), ("draw", row.draw), ("away", row.away)):
            add(row.match_id, "1x2", selection, np.nan, row.bookmaker, row.market_time, odds,
                float(forecast[f"p_{selection}"].iloc[i]))
    for row in prices["totals"].itertuples(index=False):
        i = index.get(row.match_id)
        if i is None or row.bookmaker == "max" and not args.include_max:
            continue
        add(row.match_id, "totals", "over", 2.5, row.bookmaker, row.market_time, row.over, float(forecast["p_over_2_5"].iloc[i]))
        add(row.match_id, "totals", "under", 2.5, row.bookmaker, row.market_time, row.under, float(forecast["p_under_2_5"].iloc[i]))
    for row in prices["handicap"].itertuples(index=False):
        i = index.get(row.match_id)
        if i is None or row.bookmaker == "max" and not args.include_max:
            continue
        home, away = handicap_effective_probabilities(grids[i], row.line, row.home, row.away)
        add(row.match_id, "handicap", "ah_home", row.line, row.bookmaker, row.market_time, row.home, float(home[0]))
        add(row.match_id, "handicap", "ah_away", row.line, row.bookmaker, row.market_time, row.away, float(away[0]))
    if not rows:
        listed = "\n".join(f"  {k:%a %d %b %H:%M}  {h} v {a}" for k, h, a in
                           fixtures[["kickoff", "home_team", "away_team"]].itertuples(index=False))
        raise SystemExit(
            f"No prices stored for the {len(fixtures)} fixture(s) in the next {args.upcoming} days:\n{listed}\n"
            "Football-Data publishes the week's prices in fixtures.csv a few days before kickoff; run "
            "`football-predictor ingest` once it lists them (the current fixtures.csv covers other divisions only)."
        )
    board = pd.DataFrame(rows).sort_values("ev", ascending=False)
    board["kickoff"] = pd.to_datetime(board["kickoff"]).dt.strftime("%Y-%m-%d %H:%M")
    source = forecast["source"].value_counts().to_dict() if "source" in forecast else {"model": len(forecast)}
    print(f"Fair value: {args.fair} ({source}); {len(board)} prices on {len(fixtures)} fixtures; "
          f"showing EV > {args.ev_threshold:+.2f}")
    shown = board[board["ev"] > args.ev_threshold]
    if shown.empty:
        print("No price beats the fair value by that margin.")
        print(board.head(10).round(3).to_string(index=False))
    else:
        print(shown.round(3).to_string(index=False))


METRIC_SECTIONS = {
    "Forecast quality": ["matches", "brier_model", "rps_model", "log_loss_model", "score_log_loss_model",
                         "score_hit_model", "accuracy_model", "ece_model", "brier_base_rate",
                         "brier_skill_vs_base_rate"],
    "Market-anchored forecast": ["anchored_share_market", "brier_anchored", "rps_anchored", "log_loss_anchored",
                                 "accuracy_anchored", "ece_anchored", "score_log_loss_anchored", "score_hit_anchored",
                                 "brier_over_anchored"],
    "Against the market": ["matches_with_odds", "brier_model_priced", "brier_market", "brier_skill_vs_market",
                           "brier_diff_vs_market", "brier_diff_ci_low", "brier_diff_ci_high", "rps_market",
                           "log_loss_market", "accuracy_market", "ece_market", "mean_overround"],
    "Over/under 2.5 and both teams to score": [
        "matches_with_totals_odds", "over_rate", "brier_over_model", "brier_over_market",
        "brier_over_diff_vs_market", "over_diff_ci_low", "over_diff_ci_high", "log_loss_over_model",
        "mean_overround_totals", "btts_rate", "brier_btts_model", "log_loss_btts_model"],
    "Betting": ["bets", "staked", "profit", "roi", "final_bankroll", "bankroll_growth", "max_drawdown",
                "hit_rate", "mean_odds", "mean_ev", "flat_stake_profit_units", "flat_stake_roi",
                "flat_roi_ci_low", "flat_roi_ci_high", "p_flat_roi_le_zero", "mean_clv",
                "bets_1x2", "flat_stake_roi_1x2", "clv_1x2", "bets_totals", "flat_stake_roi_totals", "clv_totals",
                "bets_handicap", "flat_stake_roi_handicap", "clv_handicap",
                "bets_home", "roi_home", "bets_draw", "roi_draw", "bets_away", "roi_away",
                "bets_over", "roi_over", "bets_under", "roi_under", "bets_ah_home", "roi_ah_home",
                "bets_ah_away", "roi_ah_away"],
}


def season_breakdown(result: BacktestResult) -> pd.DataFrame:
    """Forecast quality and flat-stake betting per season of a multi-season backtest."""
    bets = result.bets
    if not bets.empty:
        bets = bets.assign(season=[season_from_date(k) for k in pd.to_datetime(bets["kickoff"])])
    rows = []
    for season, predictions in result.predictions.groupby("season"):
        m = evaluate_forecasts(predictions, n_resamples=0)
        picked = bets[bets["season"] == season] if not bets.empty else bets
        flat = picked["units"].to_numpy(dtype=float) if len(picked) else np.array([])
        rows.append(
            {
                "season": int(season),
                "matches": len(predictions),
                "brier_model": m["brier_model"],
                "brier_market": m.get("brier_market", np.nan),
                "brier_anchored": m.get("brier_anchored", np.nan),
                "accuracy_model": m["accuracy_model"],
                "accuracy_anchored": m.get("accuracy_anchored", np.nan),
                "score_hit_model": m.get("score_hit_model", np.nan),
                "score_hit_anchored": m.get("score_hit_anchored", np.nan),
                "brier_over_model": m.get("brier_over_model", np.nan),
                "brier_over_market": m.get("brier_over_market", np.nan),
                "bets": len(picked),
                "flat_stake_roi": float(flat.mean()) if len(flat) else np.nan,
                "clv": float(picked["clv"].mean()) if len(picked) else np.nan,
            }
        )
    return pd.DataFrame(rows)


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
    if args.test_seasons:
        test_seasons = parse_seasons(args.test_seasons)
    else:
        test_seasons = [args.test_season if args.test_season is not None
                        else default_test_season(data, args.division)]
    test_season, last_season = min(test_seasons), max(test_seasons)
    start, _ = _season_window(data, test_season, args.division)
    _, end = _season_window(data, last_season, args.division)
    label = f"{test_season}" if last_season == test_season else f"{test_season}-{last_season}"

    if args.tune:
        val_start, _ = _season_window(data, test_season - args.tune_seasons, args.division)
        base = {k: v for k, v in model_kwargs(args).items() if k not in TUNING_GRID}
        scores = tune_hyperparameters(
            data[data["kickoff"] < start], TUNING_GRID, val_start, start, model_kwargs=base, divisions=divisions,
            n_jobs=args.jobs,
        )
        validated = (f"season {test_season - 1}" if args.tune_seasons == 1
                     else f"seasons {test_season - args.tune_seasons}-{test_season - 1}")
        print(f"Hyperparameters validated on {validated} (best first):")
        print(scores[list(TUNING_GRID) + ["brier_model", "log_loss_model", "score_log_loss_model"]]
              .head(8).round(5).to_string(index=False))
        args.xi = float(scores.loc[0, "xi"])
        args.xg_weight = float(scores.loc[0, "xg_weight"])

    strategy = BettingStrategy(
        ev_threshold=args.ev_threshold,
        kelly_multiplier=args.kelly,
        max_stake_fraction=args.max_stake,
        max_odds=args.max_odds,
        markets=tuple(args.markets),
        fair=args.fair,
        blend_weight=args.blend_weight,
        price=args.price,
    )
    reports = Path(args.reports_dir)
    if args.compare:
        rows = []
        for preset in FEATURE_PRESETS:
            kwargs = {**model_kwargs(args), "attack_features": FEATURE_PRESETS[preset][0],
                      "defence_features": FEATURE_PRESETS[preset][1]}
            result = run_backtest(data, partial(DixonColesModel, **kwargs), start, end, refit_days=args.refit_days,
                                  strategy=strategy, initial_bankroll=args.bankroll, divisions=divisions,
                                  anchor_price=args.anchor_price)
            rows.append({"features": preset, **result.metrics})
        table = pd.DataFrame(rows)
        columns = ["features", "brier_model", "brier_market", "brier_over_model", "brier_over_market",
                   "log_loss_model", "ece_model", "bets", "roi", "flat_stake_roi", "p_flat_roi_le_zero"]
        print(f"Feature presets on {'season' if label == str(test_season) else 'seasons'} {label} "
              f"(xi={args.xi}, xg_weight={args.xg_weight}):")
        print(table[[c for c in columns if c in table]].round(4).to_string(index=False))
        reports.mkdir(parents=True, exist_ok=True)
        table.to_csv(reports / f"comparison_{label}.csv", index=False)

    result = run_backtest(
        data, model_factory(args), start, end,
        refit_days=args.refit_days, strategy=strategy, initial_bankroll=args.bankroll, divisions=divisions,
        anchor_price=args.anchor_price,
    )
    print(f"Test {'season' if label == str(test_season) else 'seasons'} {label} ({args.division}) | "
          f"xi={args.xi} | xg_weight={args.xg_weight} | features={args.features} | "
          f"markets={','.join(args.markets)} | fair={args.fair} | price={args.price}")
    print_metrics(result.metrics)
    if last_season != test_season:
        print("  Per season")
        print(season_breakdown(result).round(4).to_string(index=False))
    title = (f"Cumulative PnL, season {test_season}/{(test_season + 1) % 100:02d}" if last_season == test_season
             else f"Cumulative PnL, seasons {test_season}/{(test_season + 1) % 100:02d} to "
                  f"{last_season}/{(last_season + 1) % 100:02d}")
    paths = save_report(result, reports, label, title=title)
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
    ingest.add_argument("--sources", nargs="+", default=["football_data", "understat", "espn"],
                        choices=source_choices)
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
                       help="divisions to train on, or 'all' (default: only --division; on real data the "
                            "Championship dominance of promoted clubs misleads their Premier League ratings)")

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

    scan = sub.add_parser("scan", help="value board: every bookmaker's price on coming fixtures vs a fair value")
    model_options(scan)
    scan.add_argument("--upcoming", type=int, default=7, metavar="DAYS", help="fixtures in the next DAYS days")
    scan.add_argument("--as-of", default=None, help="fit date (default: now)")
    scan.add_argument("--fair", choices=["anchored", "model"], default="anchored",
                      help="fair value: the market-anchored grid (default) or the model alone")
    scan.add_argument("--ev-threshold", type=float, default=0.0, help="show prices with EV above this")
    scan.add_argument("--kelly", type=float, default=0.25, help="fraction of full Kelly for stake_pct")
    scan.add_argument("--include-max", action="store_true", help="also list the best-of-market price")
    scan.set_defaults(func=cmd_scan, home=None, away=None)

    for name, help_text in (("backtest", "walk-forward backtest with EV betting"),
                            ("demo", "backtest on a simulated league (no network needed)")):
        bt = sub.add_parser(name, help=help_text)
        model_options(bt)
        bt.add_argument("--test-season", type=int, default=None, help="default: the last completed season")
        bt.add_argument("--test-seasons", default=None, metavar="RANGE",
                        help="e.g. 2021-2025: one walk-forward across several seasons, reported overall "
                             "and per season (far more matches than one season can offer)")
        bt.add_argument("--markets", nargs="+", choices=["1x2", "totals", "handicap"],
                        default=["1x2", "totals", "handicap"], help="markets the EV search may bet (default: all)")
        bt.add_argument("--fair", choices=["model", "market", "anchored", "blend"], default="model",
                        help="probability the EV is judged with: the model, the de-margined closing line "
                             "(sharp price as fair value), the market-anchored grid (the sharp line's own rates, "
                             "pricing every market consistently), or a model/market blend (default: model)")
        bt.add_argument("--blend-weight", type=float, default=0.5, help="weight on the market in --fair blend")
        bt.add_argument("--anchor-price", choices=["closing", "opening"], default="closing",
                        help="prices the market-anchored forecast is built from (default: closing)")
        bt.add_argument("--price", choices=["closing", "opening", "max"], default="closing",
                        help="price to bet at: the closing reference, the earliest listed price, or the best "
                             "price across bookmakers (default: closing)")
        bt.add_argument("--tune", "--tune-xi", dest="tune", action="store_true",
                        help="pick xi and xg_weight on the seasons before the test season")
        bt.add_argument("--tune-seasons", type=int, default=2, metavar="N",
                        help="validate on the N seasons before the test season (default: %(default)s)")
        bt.add_argument("--jobs", type=int, default=DEFAULT_JOBS, help="processes for tuning (default: %(default)s)")
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
