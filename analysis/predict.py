"""Forecast one fixture with the Poisson model, or list the clubs' strengths.

    python -m analysis.predict "Arsenal" "Chelsea"     probabilities and the score grid
    python -m analysis.predict --strengths             attack and defence strength of every club

The model is fitted on the two most recent seasons in ``data/final/matches.csv``.
Club names are spelled as in the data ("Man City", "Nott'm Forest", ...).
"""

from __future__ import annotations

import argparse
import sys

from extraction.preprocess import load_matches

from .poisson import PoissonModel


def fitted_model(prior_games: float):
    matches = load_matches()
    recent = matches[matches["season"] >= matches["season"].max() - 1]  # the last two seasons
    return PoissonModel(prior_games=prior_games).fit(recent), recent


def show_forecast(home: str, away: str, prior_games: float) -> None:
    model, recent = fitted_model(prior_games)
    unknown = [club for club in (home, away) if club not in model.strengths.index]
    if unknown:
        raise SystemExit(f"Unknown club(s): {', '.join(unknown)}. Known: {', '.join(model.strengths.index)}")
    forecast = model.predict(home, away)
    print(f"{home} v {away}  (fitted on {len(recent)} matches)")
    print(f"  expected goals  {forecast['exp_home_goals']:.2f} - {forecast['exp_away_goals']:.2f}")
    print(f"  home {forecast['p_home']:.1%}   draw {forecast['p_draw']:.1%}   away {forecast['p_away']:.1%}")
    print(f"  over 2.5 {forecast['p_over_2_5']:.1%}   both teams score {forecast['p_btts']:.1%}")
    print(f"  most likely score {forecast['likely_score']} ({forecast['p_likely_score']:.1%})\n")
    print("Correct-score probabilities (%):")
    print((model.score_grid(home, away) * 100).round(1).to_string())


def show_strengths(prior_games: float) -> None:
    model, recent = fitted_model(prior_games)
    table = model.strengths.copy()
    table["attack"] = (table["home_attack"] + table["away_attack"]) / 2
    table["defence"] = (table["home_defence"] + table["away_defence"]) / 2
    print(f"Strengths from the last two seasons, {len(recent)} matches "
          "(1.0 = league average; a lower defence number is better)")
    print(table[["games", "attack", "defence"]].sort_values("attack", ascending=False).round(2).to_string())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("home", nargs="?")
    parser.add_argument("away", nargs="?")
    parser.add_argument("--strengths", action="store_true", help="list every club's attack and defence strength")
    parser.add_argument("--prior", type=float, default=5.0, help="smoothing: extra average games per club")
    args = parser.parse_args(argv)
    if args.strengths:
        show_strengths(args.prior)
    elif args.home and args.away:
        show_forecast(args.home, args.away, args.prior)
    else:
        parser.error("give a home and an away club, or --strengths")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
