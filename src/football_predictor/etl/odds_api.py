"""The Odds API (the-odds-api.com): live 1X2 prices from many bookmakers, Pinnacle among them.

Football-Data's files no longer carry Pinnacle, and its prices arrive after
the fact, so a live version of the price-shopping rule needs the sharp price
from somewhere else, logged before kickoff. The Odds API serves the current
``h2h`` (1X2) prices of the leagues in :data:`SPORT_KEYS` for an explicit
list of bookmakers. Cost: every ten bookmakers count as one region and each
region costs one credit per league per request (empty answers are free; the
free plan has 500 credits a month), so :data:`DEFAULT_BOOKMAKERS` is Pinnacle
plus nine soft books, one credit per league. The key is read from the
``ODDS_API_KEY`` environment variable and never written anywhere.

Bet365, bwin and Interwetten are not in the API; William Hill, BetVictor and
1xBet are, under the names the warehouse already uses for them, so the live
board and the backtest agree on who is who.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable, Sequence

import pandas as pd
import requests

from ..config import USER_AGENT

log = logging.getLogger(__name__)

API_URL = "https://api.the-odds-api.com/v4"
# Football-Data division -> The Odds API sport key. The National League (EC)
# and the Scottish Championship (SC1) are not covered.
SPORT_KEYS = {
    "E0": "soccer_epl", "E1": "soccer_efl_champ", "E2": "soccer_england_league1", "E3": "soccer_england_league2",
    "SC0": "soccer_spl",
    "D1": "soccer_germany_bundesliga", "D2": "soccer_germany_bundesliga2",
    "SP1": "soccer_spain_la_liga", "SP2": "soccer_spain_segunda_division",
    "I1": "soccer_italy_serie_a", "I2": "soccer_italy_serie_b",
    "F1": "soccer_france_ligue_one", "F2": "soccer_france_ligue_two",
    "N1": "soccer_netherlands_eredivisie", "B1": "soccer_belgium_first_div", "P1": "soccer_portugal_primeira_liga",
    "T1": "soccer_turkey_super_league", "G1": "soccer_greece_super_league",
}
# The API's bookmaker key -> the warehouse's bookmaker name (Football-Data's
# where the counterparty is the same). Keys the API adds later are stored
# under their own name.
BOOKMAKERS = {
    "pinnacle": "pinnacle",
    "williamhill": "william_hill",
    "betvictor": "betvictor",
    "onexbet": "1xbet",
    "betfair_ex_uk": "betfair_exchange",
    "betfair_ex_eu": "betfair_exchange",
    "betfair_sb_uk": "betfair",
    "ladbrokes_uk": "ladbrokes",
    "coral": "coral",
    "paddypower": "paddy_power",
    "skybet": "skybet",
    "betfred_uk": "betfred",
    "betway": "betway",
    "unibet_uk": "unibet",
    "sport888": "888sport",
    "boylesports": "boylesports",
    "livescorebet": "livescore_bet",
    "virginbet": "virgin_bet",
    "betano_uk": "betano",
    "leovegas": "leovegas",
    "casumo": "casumo",
    "grosvenor": "grosvenor",
    "matchbook": "matchbook",
    "smarkets": "smarkets",
    "betsson": "betsson",
    "marathonbet": "marathonbet",
    "tipico_de": "tipico",
    "winamax_fr": "winamax",
    "winamax_de": "winamax",
    "betclic_fr": "betclic",
    "nordicbet": "nordicbet",
    "coolbet": "coolbet",
    "everygame": "everygame",
}
# Pinnacle and nine soft books, all in the UK/EU regions: one credit per league.
DEFAULT_BOOKMAKERS = ("pinnacle", "williamhill", "betvictor", "onexbet", "skybet", "paddypower", "ladbrokes_uk",
                      "coral", "betfred_uk", "betway")
# Exchanges: not a soft bookmaker (their price is a market, net of commission).
EXCHANGES = frozenset({"betfair_exchange", "matchbook", "smarkets"})
SNAPSHOT_FIELDS = ["snapshot_at", "source", "division", "event_id", "kickoff_utc", "home_name", "away_name",
                   "bookmaker", "home", "draw", "away"]


class OddsApiError(RuntimeError):
    pass


def _utc_naive(value) -> pd.Timestamp:
    """An ISO instant as a naive UTC timestamp, whole seconds."""
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_convert("UTC") if stamp.tzinfo is not None else stamp.tz_localize("UTC")
    return stamp.tz_localize(None).floor("s")


class OddsApiClient:
    """A thin client: the key goes in every request's query string and nowhere else."""

    def __init__(
        self,
        api_key: str,
        *,
        session: Any | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        base_url: str = API_URL,
    ) -> None:
        if not api_key:
            raise OddsApiError("The Odds API needs a key: set ODDS_API_KEY (free plan at the-odds-api.com)")
        self._api_key = api_key
        self.session = session or requests.Session()
        if hasattr(self.session, "headers"):
            self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = base_url.rstrip("/")
        self.quota: dict[str, int | None] = {"remaining": None, "used": None, "last": None}

    def _quota(self, headers) -> None:
        for name, header in (("remaining", "x-requests-remaining"), ("used", "x-requests-used"),
                             ("last", "x-requests-last")):
            value = headers.get(header)
            if value is not None:
                try:
                    self.quota[name] = int(float(value))
                except ValueError:
                    pass

    def get(self, path: str, **params) -> Any:
        """GET ``path`` under the API root; the decoded JSON body."""
        url = f"{self.base_url}/{path.lstrip('/')}"
        params = {"apiKey": self._api_key, **params}
        for attempt in range(self.max_retries + 1):
            response = self.session.get(url, params=params, timeout=self.timeout)
            self._quota(getattr(response, "headers", {}) or {})
            if response.status_code in (500, 502, 503, 504) and attempt < self.max_retries:
                log.warning("GET %s -> %s, retrying", path, response.status_code)
                time.sleep(2 ** (attempt + 1))
                continue
            break
        if response.status_code != 200:
            try:
                message = response.json().get("message", "")
            except Exception:  # noqa: BLE001 - any body shape
                message = (getattr(response, "text", "") or "")[:200]
            hint = {401: "the key is not valid", 429: "the plan's credits are used up or requests are too frequent",
                    404: "unknown sport key"}.get(response.status_code, "")
            raise OddsApiError(f"GET {path} -> HTTP {response.status_code}{': ' + hint if hint else ''} {message}".strip())
        return response.json()

    def sports(self, *, all_sports: bool = False) -> list[dict]:
        """Sports in season (free of charge); ``all_sports`` includes those out of season."""
        return self.get("sports", **({"all": "true"} if all_sports else {}))

    def odds(self, sport_key: str, bookmakers: Sequence[str] = DEFAULT_BOOKMAKERS, markets: str = "h2h") -> list[dict]:
        """Every upcoming event of ``sport_key`` with the listed bookmakers' current prices."""
        return self.get(f"sports/{sport_key}/odds", bookmakers=",".join(bookmakers), markets=markets,
                        oddsFormat="decimal", dateFormat="iso")


def fetch_odds(client: OddsApiClient, division: str, bookmakers: Sequence[str] = DEFAULT_BOOKMAKERS) -> list[dict]:
    """The API's events for a Football-Data division; empty when the league is out of season."""
    if division not in SPORT_KEYS:
        raise OddsApiError(f"The Odds API does not cover division {division}; covered: {', '.join(SPORT_KEYS)}")
    try:
        return client.odds(SPORT_KEYS[division], bookmakers)
    except OddsApiError as exc:
        if "HTTP 404" in str(exc):  # a league the API has nothing for right now
            log.info("odds api %s: %s", division, exc)
            return []
        raise


def parse_events(events: Iterable[dict], division: str, snapshot_at) -> pd.DataFrame:
    """One row per (event, bookmaker) with the 1X2 prices, in :data:`SNAPSHOT_FIELDS`.

    The ``h2h`` market names its outcomes after the teams and ``Draw``; a
    bookmaker without all three is skipped, as is a price of 1 or below.
    """
    stamp = _utc_naive(snapshot_at)
    rows = []
    for event in events:
        home, away = event.get("home_team"), event.get("away_team")
        if not home or not away or not event.get("commence_time"):
            continue
        kickoff = _utc_naive(event["commence_time"])
        for book in event.get("bookmakers", []):
            market = next((m for m in book.get("markets", []) if m.get("key") == "h2h"), None)
            if market is None:
                continue
            prices = {}
            for outcome in market.get("outcomes", []):
                name, price = outcome.get("name"), outcome.get("price")
                selection = "home" if name == home else "away" if name == away else "draw" if name == "Draw" else None
                if selection is not None and price is not None:
                    prices[selection] = float(price)
            if set(prices) != {"home", "draw", "away"} or min(prices.values()) <= 1.0:
                continue
            rows.append((stamp, "odds_api", division, str(event.get("id")), kickoff, home, away,
                         BOOKMAKERS.get(book.get("key"), book.get("key")), prices["home"], prices["draw"], prices["away"]))
    frame = pd.DataFrame(rows, columns=SNAPSHOT_FIELDS)
    return frame.drop_duplicates(["event_id", "bookmaker"]).reset_index(drop=True)
