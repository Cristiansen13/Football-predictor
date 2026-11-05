"""Price shopping without a model: the sharp closing line as fair value and the
bet struck wherever a bookmaker beats it.

The backtest's ``--fair market --price max`` strategy needs no forecast, so it
can be evaluated straight from the stored prices, on every match with a
closing line and for any set of bookmakers: Football-Data's market maximum
(the best price over every bookmaker it follows, listed by name or not), the
bookmakers it lists by name, the soft books alone, or the Betfair Exchange net
of commission. That answers what the model-based benchmark cannot: whether
the edge is available to one bettor with real accounts, and which bookmaker
supplies it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..etl.football_data import AGGREGATES
from .backtest import MARKET_DEMARGIN, MARKETS, SELECTIONS
from .betting import net_odds, settle_units
from .metrics import bootstrap_interval, implied_probabilities

EXCHANGE = "betfair_exchange"
SHARP = ("pinnacle", EXCHANGE)
BOOK_SETS = {
    "max": "Football-Data's market maximum: the best price over every bookmaker it follows, listed or not",
    "named": "every bookmaker Football-Data lists by name, Pinnacle and the exchange included",
    "soft": "the named bookmakers without Pinnacle and the Betfair Exchange",
    "exchange": "the Betfair Exchange alone, net of commission",
}
# Fair value: the sharp closing price, then the market average, then Bet365.
REFERENCE_ORDER = ("pinnacle", "average", "bet365")
SELECTION_MARKET = np.array([next(m for m, idx in MARKETS.items() if i in idx) for i in range(len(SELECTIONS))])
# Price columns per market, named like the dataset's best-of-market columns so
# the backtest can bet at them (``PRICE_COLUMNS["max"]``).
PRICE_NAMES = {
    "1x2": {"home": "max_home", "draw": "max_draw", "away": "max_away"},
    "totals": {"over": "max_over_2_5", "under": "max_under_2_5"},
    "handicap": {"home": "max_ah_home", "away": "max_ah_away"},
}
BOOK_NAMES = {market: {c: n.replace("max_", "book_", 1) for c, n in names.items()} for market, names in PRICE_NAMES.items()}
FAIR_NAMES = {
    "1x2": {"home": "p_home", "draw": "p_draw", "away": "p_away"},
    "totals": {"over": "p_over_2_5", "under": "p_under_2_5"},
    "handicap": {"home": "p_ah_home", "away": "p_ah_away"},
}


def resolve_books(spec: str, available: Iterable[str]) -> list[str]:
    """Bookmakers behind a set name (:data:`BOOK_SETS`) or a comma-separated list."""
    available = sorted(set(available))
    named = [b for b in available if b not in AGGREGATES]
    if spec == "max":
        return ["max"]
    if spec == "named":
        return named
    if spec == "soft":
        return [b for b in named if b not in SHARP]
    if spec == "exchange":
        return [EXCHANGE]
    books = [b.strip() for b in spec.split(",") if b.strip()]
    unknown = [b for b in books if b not in available]
    if unknown:
        raise ValueError(f"unknown bookmaker(s) {unknown}; stored: {', '.join(available)}")
    return books


def _apply_commission(frame: pd.DataFrame, columns: Sequence[str], commission: float) -> pd.DataFrame:
    if commission and "bookmaker" in frame and not frame.empty:
        frame = frame.copy()
        rows = frame["bookmaker"] == EXCHANGE
        frame.loc[rows, list(columns)] = net_odds(frame.loc[rows, list(columns)].to_numpy(dtype=float), commission)
    return frame


def net_exchange(prices: dict[str, pd.DataFrame], commission: float) -> dict[str, pd.DataFrame]:
    """The stored prices with the Betfair Exchange's netted of ``commission`` in every market."""
    out = {}
    for market, frame in prices.items():
        names = PRICE_NAMES.get(market)
        out[market] = _apply_commission(frame, list(names), commission) if names is not None and frame is not None else frame
    return out


def _filter(frame: pd.DataFrame, books: Sequence[str], market_time: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["match_id", "bookmaker", "market_time"])
    out = frame[(frame["market_time"] == market_time) & frame["bookmaker"].isin(list(books))].copy()
    rank = {b: i for i, b in enumerate(books)}
    out["_rank"] = out["bookmaker"].map(rank)
    return out.sort_values(["match_id", "_rank"], kind="stable")


def _modal_line(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep each fixture's rows at its most quoted handicap line (Football-Data quotes one)."""
    if frame.empty:
        return frame
    counts = frame.groupby(["match_id", "line"], sort=False).size().reset_index(name="n")
    chosen = counts.loc[counts.groupby("match_id")["n"].idxmax()]  # ties: the line quoted first in bookmaker order
    lines = chosen.set_index("match_id")["line"]
    return frame[frame["line"].to_numpy() == frame["match_id"].map(lines).to_numpy()]


def best_prices(
    prices: dict[str, pd.DataFrame],
    books: Sequence[str],
    *,
    market_time: str = "closing",
    commission: float = 0.0,
) -> pd.DataFrame:
    """Best price per selection over ``books`` and which bookmaker gives it, indexed by fixture.

    Ties go to the earlier bookmaker in ``books``. Exchange prices are netted
    of ``commission``. Handicap prices are compared at the fixture's quoted
    line (``max_ah_line``).
    """
    frames = []
    for market, names in PRICE_NAMES.items():
        frame = _filter(prices.get(market), books, market_time)
        frame = _apply_commission(frame, list(names), commission)
        if market == "handicap":
            frame = _modal_line(frame)
        if frame.empty:
            continue
        out = pd.DataFrame(index=pd.Index(frame["match_id"].unique(), name="match_id"))
        for column, name in names.items():
            idx = frame.groupby("match_id")[column].idxmax()
            best = frame.loc[idx].set_index("match_id")
            out[name] = best[column]
            out[BOOK_NAMES[market][column]] = best["bookmaker"]
        if market == "handicap":
            out["max_ah_line"] = frame.groupby("match_id")["line"].first()
        frames.append(out)
    if not frames:
        columns = [n for names in PRICE_NAMES.values() for n in names.values()]
        columns += [n for names in BOOK_NAMES.values() for n in names.values()] + ["max_ah_line"]
        return pd.DataFrame(columns=columns, index=pd.Index([], name="match_id"))
    return pd.concat(frames, axis=1)


def reference_probabilities(
    prices: dict[str, pd.DataFrame],
    *,
    book: str = "pinnacle",
    market_time: str = "closing",
    fallback: Sequence[str] = REFERENCE_ORDER,
) -> pd.DataFrame:
    """De-margined probabilities of ``book``'s price per market, falling back
    along ``fallback`` where it has none; ``ref_<market>`` names the source."""
    candidates = [book] + [b for b in fallback if b != book]
    frames = []
    for market, names in FAIR_NAMES.items():
        frame = _filter(prices.get(market), candidates, market_time)
        if market == "handicap":
            frame = _modal_line(frame)
        if frame.empty:
            continue
        first = frame.groupby("match_id").head(1).set_index("match_id")
        probabilities = implied_probabilities(first[list(names)].to_numpy(dtype=float), MARKET_DEMARGIN)
        out = pd.DataFrame(probabilities, index=first.index, columns=list(names.values()))
        out[f"ref_{market}"] = first["bookmaker"]
        if market == "handicap":
            out["ref_ah_line"] = first["line"]
        frames.append(out)
    if not frames:
        return pd.DataFrame(index=pd.Index([], name="match_id"))
    return pd.concat(frames, axis=1)


def summarize(bets: pd.DataFrame, *, n_resamples: int = 2000, seed: int = 0) -> dict:
    """Flat-stake return, its bootstrap interval, closing-line value and hit rate."""
    if bets.empty:
        return {"bets": 0, "flat_stake_roi": float("nan"), "mean_clv": float("nan"), "hit_rate": float("nan"),
                "mean_odds": float("nan"), "flat_roi_ci_low": float("nan"), "flat_roi_ci_high": float("nan"),
                "p_flat_roi_le_zero": float("nan")}
    units = bets["units"].to_numpy(dtype=float)
    out = {
        "bets": int(len(bets)),
        "flat_stake_roi": float(units.mean()),
        "mean_ev": float(bets["ev"].mean()),
        "mean_clv": float(bets["clv"].mean()) if "clv" in bets else float(bets["ev"].mean()),
        "hit_rate": float((units > 0).mean()),
        "mean_odds": float(bets["odds"].mean()),
    }
    if n_resamples and len(bets) > 1:
        low, high, stats = bootstrap_interval(np.mean, units, n_resamples=n_resamples, seed=seed)
        out.update({"flat_roi_ci_low": low, "flat_roi_ci_high": high, "p_flat_roi_le_zero": float((stats <= 0).mean())})
    return out


def _table(bets: pd.DataFrame, by: str, *, n_resamples: int) -> pd.DataFrame:
    rows = [{by: key, **summarize(group, n_resamples=n_resamples)} for key, group in bets.groupby(by, sort=True)]
    return pd.DataFrame(rows)


@dataclass
class ShoppingResult:
    bets: pd.DataFrame
    metrics: dict
    by_season: pd.DataFrame
    by_market: pd.DataFrame
    by_bookmaker: pd.DataFrame
    by_division: pd.DataFrame = field(default_factory=pd.DataFrame)
    coverage: dict = field(default_factory=dict)


def price_shopping(
    matches: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    *,
    books: Sequence[str],
    fair_book: str = "pinnacle",
    strict_fair: bool = False,
    markets: Sequence[str] = ("1x2", "totals", "handicap"),
    ev_threshold: float = 0.02,
    commission: float = 0.05,
    min_odds: float = 1.01,
    max_odds: float = 15.0,
    one_per_match: bool = True,
    market_time: str = "closing",
    n_resamples: int = 2000,
) -> ShoppingResult:
    """Bet wherever a bookmaker in ``books`` beats the de-margined price of ``fair_book``.

    ``matches`` needs match_id, season, kickoff, home_team, away_team and the
    goals; unplayed rows are ignored. Per fixture the highest-EV selection is
    taken (every selection above the threshold with ``one_per_match=False``),
    on flat one-unit stakes, settled at the price taken. Both the price and
    the fair value are read at ``market_time``: at closing the EV of each bet
    is also its closing-line value, since the fair value *is* the closing
    line; at ``"opening"`` (Football-Data's earlier collection, days before
    kickoff) the bet is judged by the fair bookmaker's opening price and its
    closing-line value (``clv``) by the same bookmaker's closing price. With
    ``strict_fair`` fixtures the fair bookmaker does not price are skipped
    rather than judged by the next reference price.
    """
    played = matches[matches["home_goals"].notna() & matches["away_goals"].notna()]
    played = played.sort_values("kickoff").reset_index(drop=True)
    ids = played["match_id"]
    best = best_prices(prices, books, market_time=market_time, commission=commission).reindex(ids)
    fair = reference_probabilities(prices, book=fair_book, market_time=market_time).reindex(ids)
    closing = fair if market_time == "closing" else reference_probabilities(prices, book=fair_book).reindex(ids)
    n, k = len(played), len(SELECTIONS)
    odds = np.full((n, k), np.nan)
    probability = np.full((n, k), np.nan)
    p_closing = np.full((n, k), np.nan)
    bookmaker = np.full((n, k), "", dtype=object)
    source = np.full((n, k), "", dtype=object)
    lines = np.full(n, np.nan)
    for market, idx in MARKETS.items():
        price_names, book_names, fair_names = PRICE_NAMES[market], BOOK_NAMES[market], FAIR_NAMES[market]
        for j, column in zip(idx, price_names):
            if price_names[column] in best:
                odds[:, j] = best[price_names[column]].to_numpy(dtype=float)
                bookmaker[:, j] = best[book_names[column]].fillna("").to_numpy()
            if fair_names[column] in fair:
                probability[:, j] = fair[fair_names[column]].to_numpy(dtype=float)
                source[:, j] = fair[f"ref_{market}"].fillna("").to_numpy()
            if fair_names[column] in closing:
                p_closing[:, j] = closing[fair_names[column]].to_numpy(dtype=float)
        if market == "handicap" and "max_ah_line" in best:
            lines = best["max_ah_line"].to_numpy(dtype=float)
            # A handicap bet is judged, and valued at closing, only at the same line.
            if "ref_ah_line" in fair:
                odds[~np.isclose(lines, fair["ref_ah_line"].to_numpy(dtype=float)), 5:7] = np.nan
            if "ref_ah_line" in closing:
                p_closing[~np.isclose(lines, closing["ref_ah_line"].to_numpy(dtype=float)), 5:7] = np.nan
    if strict_fair:
        probability[source != fair_book] = np.nan
    allowed = (np.isfinite(odds) & np.isfinite(probability) & (odds >= min_odds) & (odds <= max_odds)
               & np.isin(SELECTION_MARKET, list(markets))[None, :])
    ev = np.where(allowed, probability * odds - 1.0, -np.inf)
    if one_per_match:
        pick = ev.argmax(axis=1)
        rows = np.arange(n)
        chosen = ev[rows, pick] > ev_threshold
        rows, cols = rows[chosen], pick[chosen]
    else:
        rows, cols = np.nonzero(ev > ev_threshold)
    columns = ["match_id", "season", "kickoff", "home_team", "away_team"]
    if "division" in played:
        columns.insert(2, "division")
    bets = played.iloc[rows][columns].reset_index(drop=True)
    bets["market"] = SELECTION_MARKET[cols]
    bets["selection"] = SELECTIONS[cols]
    bets["line"] = np.where(cols >= 5, lines[rows], np.nan)
    bets["bookmaker"] = bookmaker[rows, cols]
    bets["fair_source"] = source[rows, cols]
    bets["odds"] = odds[rows, cols]
    bets["p_fair"] = probability[rows, cols]
    bets["ev"] = ev[rows, cols]
    bets["clv"] = p_closing[rows, cols] * odds[rows, cols] - 1.0
    bets["units"] = settle_units(
        cols, odds[rows, cols],
        home_goals=played["home_goals"].to_numpy(dtype=float)[rows],
        away_goals=played["away_goals"].to_numpy(dtype=float)[rows],
        line=lines[rows],
    )
    bets["won"] = bets["units"] > 0
    fair_1x2 = fair["ref_1x2"] if "ref_1x2" in fair else pd.Series(index=ids, dtype=object)
    coverage = {
        "matches": int(n),
        "matches_with_fair": int(np.isfinite(probability).any(axis=1).sum()),
        "share_fair_from_book": float((fair_1x2.to_numpy() == fair_book).mean()) if n else float("nan"),
        "matches_with_price": int(np.isfinite(odds).any(axis=1).sum()),
        "books": list(books),
    }
    return ShoppingResult(
        bets=bets,
        metrics=summarize(bets, n_resamples=n_resamples),
        by_season=_table(bets, "season", n_resamples=n_resamples),
        by_market=_table(bets, "market", n_resamples=n_resamples),
        by_bookmaker=_table(bets, "bookmaker", n_resamples=n_resamples),
        by_division=_table(bets, "division", n_resamples=n_resamples) if "division" in bets else pd.DataFrame(),
        coverage=coverage,
    )


def aggregate_gap(prices: dict[str, pd.DataFrame], books: Sequence[str], *, market_time: str = "closing") -> dict:
    """How far Football-Data's market maximum sits above the best named price.

    On 1X2 selections priced by both: the mean relative gap, the share of
    selections where the maximum comes from a bookmaker outside ``books``
    (strictly above every named price), and the share where it is at least
    1% better.
    """
    named = best_prices(prices, books, market_time=market_time)
    maximum = best_prices(prices, ["max"], market_time=market_time)
    columns = list(PRICE_NAMES["1x2"].values())
    if named.empty or maximum.empty or not all(c in named and c in maximum for c in columns):
        return {"selections": 0}
    both = named[columns].join(maximum[columns], lsuffix="_named", rsuffix="_max", how="inner").dropna()
    if both.empty:
        return {"selections": 0}
    best_named = both[[f"{c}_named" for c in columns]].to_numpy(dtype=float)
    best_max = both[[f"{c}_max" for c in columns]].to_numpy(dtype=float)
    ratio = best_max / best_named - 1.0
    return {
        "selections": int(ratio.size),
        "mean_gap": float(ratio.mean()),
        "share_unlisted_best": float((ratio > 1e-9).mean()),
        "share_gap_over_1pct": float((ratio > 0.01).mean()),
    }


def with_best_prices(data: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    """Replace the dataset's best-of-market price columns with ``best`` (from :func:`best_prices`).

    Fixtures absent from ``best`` lose their best price, so a backtest at
    ``--price max`` then bets only where the chosen bookmakers quote one.
    """
    out = data.copy()
    columns = [n for names in PRICE_NAMES.values() for n in names.values()] + ["max_ah_line"]
    columns += [n for names in BOOK_NAMES.values() for n in names.values()]
    aligned = best.reindex(out["match_id"])
    for column in columns:
        out[column] = aligned[column].to_numpy() if column in aligned else np.nan
    return out
