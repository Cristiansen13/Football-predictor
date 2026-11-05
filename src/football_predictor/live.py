"""Prices logged before kickoff, a price-shopping rule applied to them, and the paper-trading ledger.

The eighteen-league study (README, "Eighteen leagues") left one rule
standing: judge a 1X2 price by Pinnacle's de-margined price and bet
favourites, odds up to 3, wherever a soft bookmaker beats it. Football-Data's
files cannot run it forward: they no longer carry Pinnacle and they arrive
after the match. This module takes prices logged before kickoff from The Odds
API (Pinnacle and the soft books it covers) and from Football-Data's weekly
fixtures file (the soft books it covers and the API does not), resolves the
clubs to the warehouse's teams so each fixture can be settled from the result
loaded later, applies the rule to the snapshot and records what it would have
bet, the first time a fixture qualifies, in a ledger. The forward test
(:mod:`.evaluation.forward`) settles the ledger.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import season_from_date
from .db import Database, match_uuid
from .etl.entity_resolution import ClubMatcher, Resolution, club_tokens
from .etl.football_data import AGGREGATES
from .etl.odds_api import EXCHANGES
from .evaluation.backtest import MARKET_DEMARGIN
from .evaluation.metrics import implied_probabilities
from .evaluation.shopping import SHARP

log = logging.getLogger(__name__)

SELECTIONS = ("home", "draw", "away")
# Not a soft bookmaker: the sharp books, the exchanges and the market aggregates.
NOT_SOFT = frozenset(SHARP) | EXCHANGES | frozenset(AGGREGATES)
# Spellings no token can bridge, by Football-Data's name of the club.
SEED_ALIASES = {
    "FC Cologne": "FC Koln",
    "Stade Rennais": "Rennes",
    "Istanbul Basaksehir": "Buyuksehyr",
    "Basaksehir": "Buyuksehyr",
    "Sporting CP": "Sp Lisbon",
    "Vitória SC": "Guimaraes",
}
BOARD_COLUMNS = ["fixture", "match_id", "source", "division", "event_id", "kickoff_utc", "home_name", "away_name",
                 "selection", "bookmaker", "odds", "p_fair", "fair_odds", "ev", "snapshot_at", "fair_at"]
LEDGER_COLUMNS = ["match_id", "placed_at", "source", "division", "kickoff_utc", "home_name", "away_name", "selection",
                  "bookmaker", "odds", "fair_book", "p_fair", "ev"]


@dataclass(frozen=True)
class Rule:
    """A price-shopping rule on 1X2 prices logged before kickoff."""

    name: str = "favourites"
    fair_book: str = "pinnacle"
    max_odds: float = 3.0
    ev_threshold: float = 0.0
    books: tuple[str, ...] | None = None  # None: every soft bookmaker
    # Bet only fixtures this close to kickoff. The API lists rounds weeks ahead,
    # and the eighteen-league study found early prices lose their edge by closing.
    horizon_hours: float = 72.0

    def describe(self) -> str:
        books = "every soft bookmaker" if self.books is None else ", ".join(self.books)
        return (f"{books} against the de-margined price of {self.fair_book}, odds up to {self.max_odds:g}, "
                f"EV above {self.ev_threshold:+.1%}, the best selection per fixture, fixtures within "
                f"{self.horizon_hours:g} h of kickoff")


FAVOURITES = Rule()


def is_soft(bookmaker: str) -> bool:
    return bookmaker not in NOT_SOFT


class ClubResolver:
    """Resolve a live source's club names to the warehouse's teams, league by league.

    An alias already stored for the source is exact. A new spelling is matched
    by :class:`ClubMatcher` against the clubs with a fixture in the division's
    latest season, then against the season before (the first weeks of a season,
    before every club has played); a match is stored as an alias, so it holds
    from then on and can be corrected by hand (``log-prices --alias``). Names
    that resolve to nothing are collected with their nearest candidates.
    """

    def __init__(self, db: Database, source: str, *, seed: dict[str, str] | None = None) -> None:
        self.db = db
        self.source = source
        self.seed = {**SEED_ALIASES, **(seed or {})}
        aliases = db.query("SELECT alias, team_id FROM team_aliases WHERE source = ?", [source])
        self._aliases: dict[str, str] = dict(zip(aliases["alias"], aliases["team_id"]))
        self._matchers: dict[str, list[tuple[ClubMatcher, dict[str, str]]]] = {}
        self.unresolved: dict[tuple[str, str], list[tuple[str, float]]] = {}
        self.matched: list[tuple[str, str, str, float]] = []  # division, name, canonical, score

    def _matchers_for(self, division: str) -> list[tuple[ClubMatcher, dict[str, str]]]:
        if division not in self._matchers:
            seasons = self.db.query(
                "SELECT DISTINCT season FROM matches WHERE division = ? ORDER BY season DESC LIMIT 2", [division]
            )["season"].tolist()
            matchers = []
            for season in seasons:
                clubs = self.db.division_clubs(division, [int(season)])
                matchers.append((ClubMatcher(clubs), {team: names[0] for team, names in clubs.items()}))
            self._matchers[division] = matchers
        return self._matchers[division]

    def resolve(self, division: str, name: str) -> str | None:
        """The team id ``name`` denotes in ``division``, or None."""
        if name in self._aliases:
            return self._aliases[name]
        nearest: list[tuple[str, float]] = []
        for matcher, canonical in self._matchers_for(division):
            hit = matcher.match(name)
            if hit is None and name in self.seed:
                wanted = club_tokens(self.seed[name])
                hit = next(((team, 1.0) for team, c in canonical.items() if club_tokens(c) == wanted), None)
            if hit is not None:
                team, score = hit
                method = "alias" if score >= 1.0 else "fuzzy"
                self.db.register_team(Resolution(name, canonical[team], team, float(score), method), self.source)
                self._aliases[name] = team
                self.matched.append((division, name, canonical[team], float(score)))
                return team
            nearest = nearest or [(canonical[t], s) for t, s in matcher.candidates(name, limit=2)]
        self.unresolved[(division, name)] = nearest
        return None

    def add_alias(self, division: str, name: str, team_id: str, canonical: str) -> None:
        """Store ``name`` as the source's spelling of a club by hand."""
        self.db.register_team(Resolution(name, canonical, team_id, 1.0, "alias"), self.source)
        self._aliases[name] = team_id
        self.unresolved.pop((division, name), None)


def resolve_fixtures(frame: pd.DataFrame, resolver: ClubResolver) -> pd.DataFrame:
    """Snapshot rows with ``home_team_id``, ``away_team_id`` and ``match_id`` (None where a name is unresolved).

    The match id is the warehouse's, derived from the season of the kickoff
    and the two team ids, so it meets the fixture's row in ``matches`` once
    the result is loaded, whether or not that row exists yet.
    """
    frame = frame.copy()
    for column in ("home_name", "away_name"):
        ids = {}
        for division, name in frame[["division", column]].drop_duplicates().itertuples(index=False):
            ids[(division, name)] = resolver.resolve(division, name)
        frame[column.replace("_name", "_team_id")] = [ids[(d, n)] for d, n in zip(frame["division"], frame[column])]
    frame["match_id"] = [
        match_uuid(season_from_date(k), h, a) if isinstance(h, str) and isinstance(a, str) else None
        for k, h, a in zip(frame["kickoff_utc"], frame["home_team_id"], frame["away_team_id"])
    ]
    return frame


def _fixture_keys(snapshot: pd.DataFrame) -> pd.Series:
    """A fixture's key across sources: the match id when known, the source's own id otherwise."""
    own = snapshot["source"] + ":" + snapshot["event_id"]
    if "match_id" not in snapshot:
        return own
    return snapshot["match_id"].where(snapshot["match_id"].notna(), own)


def fair_values(snapshot: pd.DataFrame, fair_book: str) -> pd.DataFrame:
    """The fair bookmaker's latest de-margined probabilities per fixture key."""
    fair = snapshot[snapshot["bookmaker"] == fair_book]
    if fair.empty:
        return pd.DataFrame(columns=["p_home", "p_draw", "p_away", "fair_at"], index=pd.Index([], name="fixture"))
    # The latest price; between sources logged together, The Odds API's (it sorts last).
    fair = fair.assign(fixture=_fixture_keys(fair)).sort_values(["snapshot_at", "source"]).groupby("fixture").tail(1)
    probabilities = implied_probabilities(fair[["home", "draw", "away"]].to_numpy(dtype=float), MARKET_DEMARGIN)
    out = pd.DataFrame(probabilities, columns=["p_home", "p_draw", "p_away"], index=fair["fixture"].to_numpy())
    out["fair_at"] = fair["snapshot_at"].to_numpy()
    out.index.name = "fixture"
    return out


def value_board(snapshot: pd.DataFrame, rule: Rule = FAVOURITES, *, every_price: bool = False) -> pd.DataFrame:
    """Every soft bookmaker's latest price per fixture and selection against the fair value.

    Rows are the rule's qualifying prices (odds within its cap, EV above its
    threshold); ``every_price`` lists all of them. Fixtures the fair bookmaker
    does not price are left out: without the sharp line there is no value to
    judge. Sorted by kickoff, then EV.
    """
    if snapshot.empty:
        return pd.DataFrame(columns=BOARD_COLUMNS)
    frame = snapshot.assign(fixture=_fixture_keys(snapshot))
    if "match_id" not in frame:
        frame["match_id"] = None
    fair = fair_values(frame, rule.fair_book)
    books = frame[frame["bookmaker"].map(is_soft)] if rule.books is None else frame[frame["bookmaker"].isin(rule.books)]
    books = books.sort_values("snapshot_at").groupby(["fixture", "source", "bookmaker"]).tail(1)
    books = books.join(fair, on="fixture", how="inner")
    parts = []
    for selection in SELECTIONS:
        part = books[["fixture", "match_id", "source", "division", "event_id", "kickoff_utc", "home_name", "away_name",
                      "bookmaker", "snapshot_at", "fair_at"]].copy()
        part["selection"] = selection
        part["odds"] = books[selection].to_numpy(dtype=float)
        part["p_fair"] = books[f"p_{selection}"].to_numpy(dtype=float)
        parts.append(part)
    long = pd.concat(parts, ignore_index=True)
    long["fair_odds"] = 1.0 / long["p_fair"]
    long["ev"] = long["p_fair"] * long["odds"] - 1.0
    if not every_price:
        long = long[(long["odds"] <= rule.max_odds) & (long["ev"] > rule.ev_threshold)]
    return long.sort_values(["kickoff_utc", "fixture", "ev"], ascending=[True, True, False])[BOARD_COLUMNS].reset_index(drop=True)


def select_bets(snapshot: pd.DataFrame, rule: Rule = FAVOURITES) -> pd.DataFrame:
    """The rule's bet per fixture in a snapshot: its best qualifying selection, in :data:`LEDGER_COLUMNS`.

    Only fixtures with a warehouse match id are bet, since nothing else can
    be settled, and only those within the rule's horizon of kickoff. Ties go
    to the bookmaker named first alphabetically.
    """
    board = value_board(snapshot, rule)
    hours_before = (pd.to_datetime(board["kickoff_utc"]) - pd.to_datetime(board["snapshot_at"])).dt.total_seconds() / 3600
    board = board[board["match_id"].notna() & (hours_before <= rule.horizon_hours)]
    if board.empty:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    best = (board.sort_values(["match_id", "ev", "bookmaker"], ascending=[True, False, True])
            .groupby("match_id").head(1))
    return (best.rename(columns={"snapshot_at": "placed_at"}).assign(fair_book=rule.fair_book)[LEDGER_COLUMNS]
            .sort_values(["kickoff_utc", "home_name"]).reset_index(drop=True))


@dataclass
class LogResult:
    snapshot: pd.DataFrame
    bets: pd.DataFrame
    new_bets: int
    unresolved: dict[tuple[str, str], list[tuple[str, float]]] = field(default_factory=dict)
    matched: list[tuple[str, str, str, float]] = field(default_factory=list)


def log_prices(
    db: Database,
    frames: list[pd.DataFrame],
    resolvers: dict[str, ClubResolver],
    *,
    rule: Rule = FAVOURITES,
) -> LogResult:
    """Store the snapshot frames (one per source, same ``snapshot_at``), then record the rule's bets.

    A fixture already in the ledger for the rule keeps its first bet; prices
    keep accumulating, so the forward test can value every bet at the fair
    bookmaker's last price before kickoff.
    """
    resolved = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        for source, part in frame.groupby("source", sort=False):
            if source not in resolvers:
                resolvers[source] = ClubResolver(db, str(source))
            resolved.append(resolve_fixtures(part, resolvers[source]))
    if not resolved:
        return LogResult(pd.DataFrame(), pd.DataFrame(columns=LEDGER_COLUMNS), 0)
    snapshot = pd.concat(resolved, ignore_index=True)
    db.upsert_price_snapshots(snapshot)
    known = snapshot[snapshot["match_id"].notna()][["source", "event_id", "match_id"]].drop_duplicates()
    for row in known.itertuples(index=False):
        db.fill_snapshot_match_ids(row.source, row.event_id, row.match_id)
    bets = select_bets(snapshot, rule)
    new = db.insert_paper_bets(bets.assign(rule=rule.name)) if not bets.empty else 0
    unresolved, matched = {}, []
    for resolver in resolvers.values():
        unresolved.update(resolver.unresolved)
        matched.extend(resolver.matched)
    return LogResult(snapshot, bets, new, unresolved, matched)


def board(db: Database, rule: Rule = FAVOURITES, *, now, days: int = 7, every_price: bool = False) -> pd.DataFrame:
    """The value board from the latest logged prices of fixtures in the next ``days`` days.

    Adds ``age_hours`` (since the price was logged) and ``bet_at`` (when the
    ledger took the fixture, blank otherwise).
    """
    now = pd.Timestamp(now)
    snapshot = db.load_price_snapshots(kickoff_from=now, kickoff_to=now + pd.Timedelta(days=days))
    out = value_board(snapshot, rule, every_price=every_price)
    out["age_hours"] = (now - pd.to_datetime(out["snapshot_at"])).dt.total_seconds() / 3600 if len(out) else np.nan
    ledger = db.load_paper_bets(rule.name)
    placed = dict(zip(ledger["match_id"], ledger["placed_at"])) if not ledger.empty else {}
    out["bet_at"] = out["match_id"].map(placed)
    return out
