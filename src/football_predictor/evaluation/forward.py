"""The forward test: the paper-trading ledger settled against results and
valued against the fair bookmaker's last logged price before kickoff.

A backtest can only show that a rule *would have* paid; the ledger shows
what it picked before the matches were played, at prices that were there to
be taken. Each bet settles from the goals in ``matches`` once the result is
loaded (``ingest`` / ``make data-leagues``), and its closing-line value is
read from the last price of the fair bookmaker logged before kickoff, so the
test reports both the return and whether the line moved the bet's way. A
third question the backtest could never answer is whether the price was
still there: each bet is also compared with its own bookmaker's last price
logged before kickoff, when a later snapshot of it exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..db import Database
from .backtest import MARKET_DEMARGIN
from .betting import settle_units
from .metrics import implied_probabilities
from .shopping import summarize, summarize_by

SELECTION_INDEX = {"home": 0, "draw": 1, "away": 2}


@dataclass
class ForwardResult:
    bets: pd.DataFrame  # every ledger row with status, units, p_close, clv, closing_at, closing_lead_hours,
    # last_odds, last_odds_at, price_held
    settled: pd.DataFrame
    pending: pd.DataFrame
    metrics: dict
    by_division: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_bookmaker: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_selection: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_source: pd.DataFrame = field(default_factory=pd.DataFrame)


def closing_values(db: Database, bets: pd.DataFrame, fair_book: str) -> pd.DataFrame:
    """Per bet: the fair bookmaker's last logged probability of the selection before kickoff and when it was logged."""
    out = pd.DataFrame({"p_close": np.nan, "closing_at": pd.NaT}, index=bets.index)
    if bets.empty:
        return out
    prices = db.load_price_snapshots(match_ids=bets["match_id"], bookmaker=fair_book)
    if prices.empty:
        return out
    kickoff = bets.set_index("match_id")["kickoff_utc"]
    prices = prices[prices["snapshot_at"] <= prices["match_id"].map(kickoff)]
    last = prices.sort_values("snapshot_at").groupby("match_id").tail(1).set_index("match_id")
    if last.empty:
        return out
    probabilities = implied_probabilities(last[["home", "draw", "away"]].to_numpy(dtype=float), MARKET_DEMARGIN)
    table = pd.DataFrame(probabilities, columns=["home", "draw", "away"], index=last.index)
    for i, row in bets.iterrows():
        if row.match_id in table.index:
            out.loc[i, "p_close"] = table.loc[row.match_id, row.selection]
            out.loc[i, "closing_at"] = last.loc[row.match_id, "snapshot_at"]
    return out


def last_prices(db: Database, bets: pd.DataFrame) -> pd.DataFrame:
    """Per bet: its bookmaker's last logged price of the selection before kickoff and when it was logged.

    The bet's own snapshot counts, so a bet with no later snapshot of its
    bookmaker reads its own price back; ``forward_test`` tells the two apart
    by the time.
    """
    out = pd.DataFrame({"last_odds": np.nan, "last_odds_at": pd.NaT}, index=bets.index)
    if bets.empty:
        return out
    prices = db.load_price_snapshots(match_ids=bets["match_id"])
    if prices.empty:
        return out
    kickoff = bets.set_index("match_id")["kickoff_utc"]
    prices = prices[prices["snapshot_at"] <= prices["match_id"].map(kickoff)]
    last = (prices.sort_values("snapshot_at").groupby(["match_id", "bookmaker"]).tail(1)
            .set_index(["match_id", "bookmaker"]))
    for i, row in bets.iterrows():
        key = (row.match_id, row.bookmaker)
        if key in last.index:
            out.loc[i, "last_odds"] = float(last.loc[key, row.selection])
            out.loc[i, "last_odds_at"] = last.loc[key, "snapshot_at"]
    return out


def forward_test(db: Database, rule: str = "favourites", *, fair_book: str | None = None,
                 n_resamples: int = 2000) -> ForwardResult:
    """Settle the ledger of ``rule`` and summarise it like a backtest."""
    bets = db.load_paper_bets(rule)
    if bets.empty:
        empty = pd.DataFrame()
        return ForwardResult(empty, empty, empty, {"recorded": 0, "settled": 0, "pending": 0, **summarize(empty)})
    fair_book = fair_book or str(bets["fair_book"].iloc[0])
    closing = closing_values(db, bets, fair_book)
    bets = bets.assign(p_close=closing["p_close"].to_numpy(), closing_at=pd.to_datetime(closing["closing_at"]))
    bets["clv"] = bets["p_close"] * bets["odds"] - 1.0
    bets["closing_lead_hours"] = (bets["kickoff_utc"] - bets["closing_at"]).dt.total_seconds() / 3600
    # Was the price still there? Only a snapshot of the bookmaker taken after the bet can say.
    last = last_prices(db, bets)
    bets["last_odds"] = last["last_odds"].to_numpy(dtype=float)
    bets["last_odds_at"] = pd.to_datetime(last["last_odds_at"])
    checked = (bets["last_odds_at"] > bets["placed_at"]).fillna(False)
    bets["price_held"] = pd.Series(pd.NA, index=bets.index, dtype="boolean")
    bets.loc[checked, "price_held"] = (bets.loc[checked, "last_odds"] >= bets.loc[checked, "odds"] - 1e-9).to_numpy()
    played = bets["home_goals"].notna() & bets["away_goals"].notna()
    units = np.full(len(bets), np.nan)
    if played.any():
        rows = played.to_numpy()
        units[rows] = settle_units(
            bets.loc[played, "selection"].map(SELECTION_INDEX).to_numpy(), bets.loc[played, "odds"].to_numpy(dtype=float),
            home_goals=bets.loc[played, "home_goals"].to_numpy(dtype=float),
            away_goals=bets.loc[played, "away_goals"].to_numpy(dtype=float),
        )
    bets["units"] = units
    bets["won"] = bets["units"] > 0
    bets["status"] = np.where(played, np.where(bets["won"], "won", "lost"), "pending")
    settled = bets[played].reset_index(drop=True)
    pending = bets[~played].reset_index(drop=True)
    metrics = {
        "recorded": int(len(bets)),
        "settled": int(len(settled)),
        "pending": int(len(pending)),
        **summarize(settled, n_resamples=n_resamples),
        "mean_clv_all": float(bets["clv"].mean()) if bets["clv"].notna().any() else float("nan"),
        "mean_closing_lead_hours": float(bets["closing_lead_hours"].mean()) if bets["closing_at"].notna().any() else float("nan"),
        "share_with_closing_move": float(((bets["closing_at"] - bets["placed_at"]).dt.total_seconds() > 0).mean()),
        # Availability: among the bets with a later snapshot of their bookmaker, how often the price taken (or a
        # better one) was still there at the last one, and how far that last price sat from the one taken.
        "price_checked": int(checked.sum()),
        "share_price_held": float(bets.loc[checked, "price_held"].astype(float).mean()) if checked.any() else float("nan"),
        "mean_price_drift": (float((bets.loc[checked, "last_odds"] / bets.loc[checked, "odds"] - 1.0).mean())
                             if checked.any() else float("nan")),
    }
    tables = {by: summarize_by(settled, by, n_resamples=n_resamples) if len(settled) else pd.DataFrame()
              for by in ("division", "bookmaker", "selection", "source")}
    return ForwardResult(bets, settled, pending, metrics, tables["division"], tables["bookmaker"], tables["selection"],
                         tables["source"])
