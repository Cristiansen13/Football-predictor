"""The Odds API client and parser."""

import pandas as pd
import pytest

from football_predictor.etl.odds_api import (
    DEFAULT_BOOKMAKERS,
    SPORT_KEYS,
    OddsApiClient,
    OddsApiError,
    fetch_odds,
    parse_events,
)

from . import fake_sources as fake


def test_parse_events_reads_the_h2h_market_under_warehouse_names():
    frame = parse_events(fake.odds_api_events("E0"), "E0", "2023-09-14T10:00:00Z")
    assert len(frame) == 6 and set(frame["bookmaker"]) == {"pinnacle", "william_hill", "betvictor"}
    assert (frame["source"] == "odds_api").all() and (frame["division"] == "E0").all()
    assert frame["snapshot_at"].drop_duplicates().tolist() == [pd.Timestamp("2023-09-14 10:00")]  # naive UTC
    row = frame[(frame["home_name"] == "Wolverhampton Wanderers") & (frame["bookmaker"] == "william_hill")].iloc[0]
    assert (row["home"], row["draw"], row["away"]) == (3.10, 3.50, 2.50)  # Draw is the outcome named "Draw"
    assert row["kickoff_utc"] == pd.Timestamp("2023-09-16 14:00") and row["away_name"] == "Manchester United"
    assert row["event_id"] == "evtE00"


def test_parse_events_skips_incomplete_and_impossible_prices():
    events = fake.odds_api_events("E0")
    events[0]["bookmakers"][1]["markets"][0]["outcomes"].pop()  # William Hill without a draw price
    events[0]["bookmakers"][2]["markets"][0]["outcomes"][0]["price"] = 1.0  # a "price" of 1
    events[1]["bookmakers"][0]["markets"] = [{"key": "totals", "outcomes": []}]  # Pinnacle without a 1X2 market
    events.append({"id": "x", "commence_time": None, "home_team": "A", "away_team": "B", "bookmakers": []})
    frame = parse_events(events, "E0", pd.Timestamp("2023-09-14 10:00"))
    assert frame.groupby("event_id")["bookmaker"].apply(set).to_dict() == {
        "evtE00": {"pinnacle"}, "evtE01": {"william_hill", "betvictor"}}
    assert parse_events([], "E0", "2023-09-14T10:00:00Z").empty


def test_client_sends_the_key_and_the_bookmakers_and_reads_the_quota():
    session = fake.FakeOddsSession()
    client = OddsApiClient("test-key", session=session)
    events = fetch_odds(client, "E0")
    assert len(events) == 2 and client.quota == {"remaining": 499, "used": 1, "last": 1}
    url, params = session.requests[-1]
    assert url.endswith(f"/sports/{SPORT_KEYS['E0']}/odds") and params["apiKey"] == "test-key"
    assert params["bookmakers"] == ",".join(DEFAULT_BOOKMAKERS) and params["markets"] == "h2h"
    assert params["oddsFormat"] == "decimal"
    assert fetch_odds(client, "D1") == []  # a league the API has nothing for: HTTP 404, no credit
    assert client.quota["remaining"] == 499
    assert [s["key"] for s in client.sports()] == list(fake.ODDS_API_SPORTS)


def test_client_errors_name_the_cause_and_never_the_key():
    with pytest.raises(OddsApiError, match="ODDS_API_KEY"):
        OddsApiClient("")
    client = OddsApiClient("wrong-key", session=fake.FakeOddsSession())
    with pytest.raises(OddsApiError) as failed:
        fetch_odds(client, "E0")
    assert "HTTP 401" in str(failed.value) and "not valid" in str(failed.value) and "wrong-key" not in str(failed.value)
    with pytest.raises(OddsApiError, match="does not cover division EC"):
        fetch_odds(OddsApiClient("test-key", session=fake.FakeOddsSession()), "EC")
