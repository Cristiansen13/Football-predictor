"""Command line for the school project.

    python main.py fetch --seasons 2019-2025        download the season files
    python main.py backtest --test-season 2024      walk-forward evaluation of one season
    python main.py predict "Arsenal" "Chelsea"       probabilities and score grid for a fixture
    python main.py table                             team strengths from all loaded data
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from footy import PoissonModel, backtest, download_season, load_seasons, summarize

DEFAULT_SEASONS = "2019-2025"


def parse_seasons(text: str) -> list[int]:
    first, last = (int(x) for x in text.split("-")) if "-" in text else (int(text), int(text))
    return list(range(first, last + 1))


def cmd_fetch(args: argparse.Namespace) -> None:
    for season in parse_seasons(args.seasons):
        path = download_season(season)
        print(f"{season}/{(season + 1) % 100:02d}: {path}")


def cmd_backtest(args: argparse.Namespace) -> None:
    matches = load_seasons(parse_seasons(args.seasons))
    predictions = backtest(matches, args.test_season, train_seasons_back=args.train_seasons, prior_games=args.prior)
    report = summarize(predictions, threshold=args.ev_threshold)
    print(f"Season {args.test_season}/{(args.test_season + 1) % 100:02d}: {report['matches']} matches, "
          f"model refitted before every match day on the previous {args.train_seasons} season(s) and the season so far\n")
    print(f"  {'':<22}{'model':>10}{'bookmaker':>12}")
    for key, label in (("accuracy", "accuracy"), ("brier", "Brier score"), ("log_loss", "log loss")):
        book = report.get(f"{key}_bookmaker", float("nan"))
        print(f"  {label:<22}{report[f'{key}_model']:>10.3f}{book:>12.3f}")
    print(f"  {'exact score hit':<22}{report['score_hit_model']:>10.3f}{'':>12}   (always saying 1-1: {report['always_1_1_hit']:.3f})")
    print(f"  {'always home win':<22}{report['always_home_accuracy']:>10.3f}")
    print(f"\n  value bets (EV > {args.ev_threshold:.0%} at {report.get('bookmaker_odds_used', 'odds')} odds): "
          f"{report['bets']} bets, {report['bets_won']} won, flat-stake ROI {report['flat_stake_roi']:+.1%}, "
          f"{report['flat_stake_profit_units']:+.1f} units")
    if args.save:
        predictions.to_csv(args.save, index=False)
        print(f"\n  predictions written to {args.save}")


def cmd_predict(args: argparse.Namespace) -> None:
    matches = load_seasons(parse_seasons(args.seasons))
    recent = matches[matches["season"] >= matches["season"].max() - 1]  # this season and the last
    model = PoissonModel(prior_games=args.prior).fit(recent)
    forecast = model.predict(args.home, args.away)
    print(f"{args.home} v {args.away}  (fitted on {len(recent)} matches)")
    print(f"  expected goals  {forecast['exp_home_goals']:.2f} - {forecast['exp_away_goals']:.2f}")
    print(f"  home {forecast['p_home']:.1%}   draw {forecast['p_draw']:.1%}   away {forecast['p_away']:.1%}")
    print(f"  over 2.5 {forecast['p_over_2_5']:.1%}   both teams score {forecast['p_btts']:.1%}")
    print(f"  most likely score {forecast['likely_score']} ({forecast['p_likely_score']:.1%})\n")
    print("Correct-score probabilities (%):")
    print((model.score_grid(args.home, args.away) * 100).round(1).to_string())


def cmd_table(args: argparse.Namespace) -> None:
    matches = load_seasons(parse_seasons(args.seasons))
    model = PoissonModel(prior_games=args.prior).fit(matches[matches["season"] == matches["season"].max()])
    table = model.strengths.copy()
    table["attack"] = (table["home_attack"] + table["away_attack"]) / 2
    table["defence"] = (table["home_defence"] + table["away_defence"]) / 2
    print(f"Strengths from season {matches['season'].max()} (1.0 = league average; lower defence is better)")
    print(table[["games", "attack", "defence"]].sort_values("attack", ascending=False).round(2).to_string())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    shared = argparse.ArgumentParser(add_help=False)  # options every command accepts, after the command name
    shared.add_argument("--seasons", default=DEFAULT_SEASONS, help="season start years, e.g. 2019-2025")
    shared.add_argument("--prior", type=float, default=5.0, help="smoothing: extra average games per team")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="download the season files", parents=[shared]).set_defaults(func=cmd_fetch)
    bt = sub.add_parser("backtest", help="walk-forward evaluation of one season", parents=[shared])
    bt.add_argument("--test-season", type=int, default=2024)
    bt.add_argument("--train-seasons", type=int, default=2, help="previous seasons used for fitting")
    bt.add_argument("--ev-threshold", type=float, default=0.05)
    bt.add_argument("--save", default=None, help="write the predictions to this CSV")
    bt.set_defaults(func=cmd_backtest)
    pr = sub.add_parser("predict", help="probabilities and score grid for one fixture", parents=[shared])
    pr.add_argument("home")
    pr.add_argument("away")
    pr.set_defaults(func=cmd_predict)
    sub.add_parser("table", help="team strengths", parents=[shared]).set_defaults(func=cmd_table)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
