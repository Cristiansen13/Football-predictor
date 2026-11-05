"""Live price snapshots, club matching across sources, the rule's ledger and the forward test."""

import numpy as np
import pandas as pd
import pytest

from football_predictor.db import Database
from football_predictor.etl.entity_resolution import ClubMatcher, club_tokens, team_uuid
from football_predictor.etl.football_data import parse_fixtures_snapshot
from football_predictor.etl.odds_api import parse_events
from football_predictor.etl.pipeline import Pipeline
from football_predictor.evaluation.backtest import MARKET_DEMARGIN
from football_predictor.evaluation.forward import forward_test
from football_predictor.evaluation.metrics import implied_probabilities
from football_predictor.live import FAVOURITES, ClubResolver, Rule, board, log_prices, select_bets, value_board

from . import fake_sources as fake

T1 = pd.Timestamp("2023-09-14 10:00")  # two days before kickoff
T2 = pd.Timestamp("2023-09-16 13:00")  # an hour before kickoff: Pinnacle has moved


def test_club_tokens_drop_what_sources_disagree_on():
    assert club_tokens("1. FC Köln") == ["koln"]
    assert club_tokens("M'gladbach") == ["gladbach"]
    assert club_tokens("Paris Saint Germain") == ["paris", "st", "germain"]
    assert club_tokens("Nott'm Forest") == ["nottingham", "forest"]
    assert club_tokens("Atlético Madrid") == ["ath", "madrid"]
    assert club_tokens("Sporting Lisbon") == ["sp", "lisbon"]


@pytest.mark.parametrize(
    ("league", "query", "expected"),
    [
        (["Dortmund", "M'gladbach", "Ein Frankfurt", "Leverkusen", "Bayern Munich", "FC Koln", "St Pauli", "Stuttgart"],
         "Borussia Dortmund", "Dortmund"),
        (["Dortmund", "M'gladbach", "Ein Frankfurt", "Leverkusen", "Bayern Munich", "FC Koln", "St Pauli", "Stuttgart"],
         "Borussia Monchengladbach", "M'gladbach"),
        (["Dortmund", "M'gladbach", "Ein Frankfurt", "Leverkusen", "Bayern Munich", "FC Koln", "St Pauli", "Stuttgart"],
         "Bayer Leverkusen", "Leverkusen"),
        (["Dortmund", "M'gladbach", "Ein Frankfurt", "Leverkusen", "Bayern Munich", "FC Koln", "St Pauli", "Stuttgart"],
         "1. FC Köln", "FC Koln"),
        (["Dortmund", "M'gladbach", "Ein Frankfurt", "Leverkusen", "Bayern Munich", "FC Koln", "St Pauli", "Stuttgart"],
         "FC St. Pauli", "St Pauli"),
        (["Dortmund", "M'gladbach", "Ein Frankfurt", "Leverkusen", "Bayern Munich", "FC Koln", "St Pauli", "Stuttgart"],
         "FC Cologne", None),
        (["Ath Madrid", "Ath Bilbao", "Real Madrid", "Sociedad", "Betis", "Oviedo"], "Atlético Madrid", "Ath Madrid"),
        (["Ath Madrid", "Ath Bilbao", "Real Madrid", "Sociedad", "Betis", "Oviedo"], "Athletic Bilbao", "Ath Bilbao"),
        (["Ath Madrid", "Ath Bilbao", "Real Madrid", "Sociedad", "Betis", "Oviedo"], "Real Sociedad", "Sociedad"),
        (["Ath Madrid", "Ath Bilbao", "Real Madrid", "Sociedad", "Betis", "Oviedo"], "Real Betis", "Betis"),
        (["Ath Madrid", "Ath Bilbao", "Real Madrid", "Sociedad", "Betis", "Oviedo"], "Real Madrid", "Real Madrid"),
        (["Inter", "Milan", "Verona", "Roma"], "Inter Milan", "Inter"),
        (["Inter", "Milan", "Verona", "Roma"], "AC Milan", "Milan"),
        (["Inter", "Milan", "Verona", "Roma"], "Hellas Verona", "Verona"),
        (["Paris SG", "Paris FC", "Lyon", "Rennes", "Brest"], "Paris Saint Germain", "Paris SG"),
        (["Paris SG", "Paris FC", "Lyon", "Rennes", "Brest"], "Paris FC", "Paris FC"),
        (["Paris SG", "Paris FC", "Lyon", "Rennes", "Brest"], "Olympique Lyonnais", "Lyon"),
        (["Paris SG", "Paris FC", "Lyon", "Rennes", "Brest"], "Stade Brestois", "Brest"),
        (["Paris SG", "Paris FC", "Lyon", "Rennes", "Brest"], "Stade Rennais", None),
        (["Feyenoord", "Sparta Rotterdam", "For Sittard", "Ajax"], "Feyenoord Rotterdam", "Feyenoord"),
        (["Feyenoord", "Sparta Rotterdam", "For Sittard", "Ajax"], "Sparta Rotterdam", "Sparta Rotterdam"),
        (["Feyenoord", "Sparta Rotterdam", "For Sittard", "Ajax"], "Fortuna Sittard", "For Sittard"),
        (["Club Brugge", "Cercle Brugge", "St. Gilloise", "St Truiden", "Genk", "Gent"], "Club Brugge KV", "Club Brugge"),
        (["Club Brugge", "Cercle Brugge", "St. Gilloise", "St Truiden", "Genk", "Gent"], "Cercle Brugge KSV", "Cercle Brugge"),
        (["Club Brugge", "Cercle Brugge", "St. Gilloise", "St Truiden", "Genk", "Gent"], "Union Saint-Gilloise", "St. Gilloise"),
        (["Club Brugge", "Cercle Brugge", "St. Gilloise", "St Truiden", "Genk", "Gent"], "Sint-Truiden", "St Truiden"),
        (["Club Brugge", "Cercle Brugge", "St. Gilloise", "St Truiden", "Genk", "Gent"], "KRC Genk", "Genk"),
        (["Hearts", "Hibernian", "Dundee", "Dundee United", "Rangers"], "Heart of Midlothian", "Hearts"),
        (["Hearts", "Hibernian", "Dundee", "Dundee United", "Rangers"], "Dundee United", "Dundee United"),
        (["Hearts", "Hibernian", "Dundee", "Dundee United", "Rangers"], "Dundee", "Dundee"),
        (["Sheffield United", "Sheffield Wednesday", "Hull City", "Bristol City", "Leicester City"], "Sheffield Wednesday",
         "Sheffield Wednesday"),
        (["Sheffield United", "Sheffield Wednesday", "Hull City", "Bristol City", "Leicester City"], "Hull City", "Hull City"),
        (["Olympiakos", "PAOK", "AEK", "Aris"], "Olympiacos Piraeus", "Olympiakos"),
        (["Olympiakos", "PAOK", "AEK", "Aris"], "PAOK Thessaloniki", "PAOK"),
        (["Espanol", "Celta", "Getafe"], "Espanyol", "Espanol"),
        (["Ceuta", "Cadiz", "Burgos"], "Celta Fortuna", None),  # a letter apart, two different clubs
    ],
)
def test_club_matcher_bridges_the_spellings_of_two_sources(league, query, expected):
    matcher = ClubMatcher({name: [name] for name in league})
    hit = matcher.match(query)
    assert (hit[0] if hit else None) == expected


def test_club_matcher_breaks_a_tie_for_the_exact_spelling():
    # Football-Data registered the club twice; the spelling it uses now wins, and the old one still resolves.
    matcher = ClubMatcher({"old": ["Mechelen"], "new": ["KV Mechelen"]})
    assert matcher.match("KV Mechelen")[0] == "new" and matcher.match("Mechelen")[0] == "old"
    assert matcher.match("Kortrijk") is None
    # Aliases count as the club's spellings too.
    matcher = ClubMatcher({"w": ["Wolverhampton Wanderers", "Wolves"], "u": ["Manchester United", "Man United"]})
    assert matcher.match("Wolves")[0] == "w" and matcher.match("Man Utd")[0] == "u"


@pytest.fixture()
def warehouse(tmp_path):
    """The toy leagues loaded from Football-Data with this week's fixtures: clubs registered, nothing logged yet."""
    db = Database(tmp_path / "football.db")
    pipeline = Pipeline(db, fake.FakeClient(), current_season=fake.SEASON, divisions=("E0", "SC0"))
    assert pipeline.run([fake.SEASON], ["football_data"], upcoming=True) == []
    yield db
    db.close()


def _snapshot_frames(when, *, closing=False):
    return [parse_events(fake.odds_api_events("E0", closing=closing), "E0", when),
            parse_events(fake.odds_api_events("SC0", closing=closing), "SC0", when),
            parse_fixtures_snapshot(fake.fixtures_csv(), ("E0", "SC0"), when)]


def _expected_ev(prices, selection, odds):
    p = implied_probabilities(np.array([prices]), MARKET_DEMARGIN)[0]
    return p[["home", "draw", "away"].index(selection)] * odds - 1.0


def test_log_prices_resolves_clubs_across_sources_and_records_the_rule_bets(warehouse):
    db = warehouse
    resolvers = {s: ClubResolver(db, s) for s in ("odds_api", "football_data")}
    result = log_prices(db, _snapshot_frames(T1), resolvers)
    snapshot = result.snapshot
    # 3 fixtures x 3 API bookmakers, plus Football-Data's Bet365, Pinnacle and average on the same fixtures.
    assert len(snapshot) == 18 and snapshot["match_id"].notna().all() and not result.unresolved
    upcoming = set(db.query("SELECT match_id FROM matches WHERE home_goals IS NULL")["match_id"])
    assert set(snapshot["match_id"]) == upcoming and snapshot["match_id"].nunique() == 3
    # The API's spellings are stored as aliases of the warehouse's clubs, under the source's name.
    aliases = db.query("SELECT alias, team_id, method FROM team_aliases WHERE source = 'odds_api'").set_index("alias")
    assert aliases.loc["Wolverhampton Wanderers", "team_id"] == team_uuid("Wolverhampton Wanderers")
    assert aliases.loc["Manchester United", "team_id"] == team_uuid("Manchester United")
    assert aliases.loc["Rangers", "team_id"] == team_uuid("Rangers") and aliases.loc["Rangers", "method"] == "alias"
    stored = db.load_price_snapshots()
    assert len(stored) == 18 and set(stored["source"]) == {"odds_api", "football_data"}
    assert stored["snapshot_at"].drop_duplicates().tolist() == [T1]

    # The rule: soft books against Pinnacle, favourites only, the best selection per fixture.
    bets = result.bets.set_index("home_name")
    assert len(bets) == 3 and result.new_bets == 3
    assert bets.loc["Wolverhampton Wanderers", ["selection", "bookmaker", "odds"]].tolist() == ["away", "william_hill", 2.50]
    assert bets.loc["Nottingham Forest", ["selection", "bookmaker", "odds"]].tolist() == ["away", "betvictor", 1.40]
    assert bets.loc["Dundee", ["selection", "bookmaker", "odds"]].tolist() == ["home", "betvictor", 1.95]
    assert bets.loc["Dundee", "ev"] == pytest.approx(_expected_ev((1.82, 3.85, 4.30), "home", 1.95))
    assert (bets["ev"] > 0).all() and (bets["placed_at"] == T1).all() and (bets["fair_book"] == "pinnacle").all()
    # Football-Data's Bet365 prices are shopped too, but never beat the API's books here; its average is not a bookmaker.
    every = value_board(snapshot, every_price=True)
    assert "bet365" in set(every["bookmaker"]) and "average" not in set(every["bookmaker"])
    assert not value_board(snapshot, Rule(max_odds=1.5)).empty  # only Man City at 1.40 is that short
    assert select_bets(snapshot, Rule(max_odds=1.5))["home_name"].tolist() == ["Nottingham Forest"]
    # Two days before kickoff is inside the default horizon and outside a one-day one.
    assert select_bets(snapshot, Rule(horizon_hours=24)).empty and len(select_bets(snapshot, Rule(horizon_hours=60))) == 3

    # A later snapshot nearer kickoff: prices accumulate, the ledger keeps the first bet per fixture.
    later = log_prices(db, _snapshot_frames(T2, closing=True), resolvers)
    assert later.new_bets == 0 and len(later.bets) == 3 and len(db.load_price_snapshots()) == 36
    ledger = db.load_paper_bets("favourites")
    assert len(ledger) == 3 and (ledger["placed_at"] == T1).all() and ledger["home_goals"].isna().all()
    assert ledger["rule"].unique().tolist() == ["favourites"]

    # The board reads the latest price per bookmaker and marks what the ledger holds.
    table = board(db, now=T2 + pd.Timedelta(minutes=30), days=7)
    assert table["match_id"].nunique() == 3 and (table["bet_at"] == T1).all() and (table["age_hours"] == 0.5).all()
    assert (pd.to_datetime(table["fair_at"]) == T2).all()  # judged by Pinnacle's latest price
    assert len(table) > 3  # at the later price a second bookmaker beats it on one fixture
    assert board(db, now=pd.Timestamp("2023-09-17"), days=7).empty  # nothing left to come


def test_log_prices_keeps_unresolved_fixtures_out_of_the_ledger_until_taught(warehouse):
    db = warehouse
    events = fake.odds_api_events("SC0")
    events[0]["home_team"] = "Dundee FC"
    for book in events[0]["bookmakers"]:
        book["markets"][0]["outcomes"][0]["name"] = "Dundee FC"
    events[0]["bookmakers"][0]["markets"][0]["outcomes"][1]["name"] = "Glasgow Rangers"
    events[0]["away_team"] = "Glasgow Rangers"
    for book in events[0]["bookmakers"][1:]:
        book["markets"][0]["outcomes"][1]["name"] = "Glasgow Rangers"
    frame = parse_events(events, "SC0", T1)
    resolver = ClubResolver(db, "odds_api")
    result = log_prices(db, [frame], {"odds_api": resolver})
    # "Dundee FC" is Dundee, not Dundee United; "Glasgow Rangers" is Rangers: the matcher bridges both.
    assert not result.unresolved and result.new_bets == 1
    # A name nothing bridges stays unresolved: logged, listed, not bet.
    events = fake.odds_api_events("E0")
    events[0]["home_team"] = "Molineux Old Gold"
    for book in events[0]["bookmakers"]:
        book["markets"][0]["outcomes"][0]["name"] = events[0]["home_team"]
    result = log_prices(db, [parse_events(events, "E0", T1)], {"odds_api": resolver})
    assert list(result.unresolved) == [("E0", "Molineux Old Gold")]
    assert result.snapshot["match_id"].isna().sum() == 3 and result.new_bets == 1  # Forest v City is still bet
    # Taught by hand, the fixture resolves and its earlier snapshot rows get the match id.
    resolver.add_alias("E0", "Molineux Old Gold", team_uuid("Wolverhampton Wanderers"), "Wolverhampton Wanderers")
    result = log_prices(db, [parse_events(events, "E0", T2)], {"odds_api": resolver})
    assert result.snapshot["match_id"].notna().all() and result.new_bets == 1
    assert db.load_price_snapshots()["match_id"].notna().all()


def _settle(db, results):
    """Results for the three upcoming fixtures: (home Football-Data name, away, home goals, away goals)."""
    matches = db.query("SELECT m.*, h.canonical_name AS home, a.canonical_name AS away FROM matches m "
                       "JOIN teams h ON h.team_id = m.home_team_id JOIN teams a ON a.team_id = m.away_team_id "
                       "WHERE m.home_goals IS NULL")
    rows = []
    for home, away, hg, ag in results:
        row = matches[(matches["home"] == home) & (matches["away"] == away)].iloc[0]
        rows.append({"season": row["season"], "division": row["division"], "kickoff": row["kickoff"],
                     "home_team_id": row["home_team_id"], "away_team_id": row["away_team_id"],
                     "home_goals": hg, "away_goals": ag})
    db.upsert_matches(pd.DataFrame(rows), "football_data")


def test_forward_test_settles_the_ledger_and_values_it_at_the_last_sharp_price(warehouse):
    db = warehouse
    resolvers = {s: ClubResolver(db, s) for s in ("odds_api", "football_data")}
    log_prices(db, _snapshot_frames(T1), resolvers)
    before = forward_test(db, n_resamples=10)
    assert before.metrics["recorded"] == 3 and before.metrics["pending"] == 3 and before.metrics["settled"] == 0
    assert before.metrics["bets"] == 0 and np.isnan(before.metrics["flat_stake_roi"])
    assert (before.bets["status"] == "pending").all()
    # Without a later Pinnacle price the closing value is the price the bet was judged by.
    assert before.bets["clv"].to_numpy() == pytest.approx(before.bets["ev"].to_numpy())
    assert before.metrics["share_with_closing_move"] == 0.0

    log_prices(db, _snapshot_frames(T2, closing=True), resolvers)
    _settle(db, [("Wolverhampton Wanderers", "Manchester United", 0, 2),  # away won: the bet pays 1.50
                 ("Nottingham Forest", "Manchester City", 1, 1),  # draw: the away bet loses
                 ("Dundee", "Rangers", 2, 0)])  # home won: pays 0.95
    result = forward_test(db, n_resamples=50)
    m = result.metrics
    assert (m["recorded"], m["settled"], m["pending"]) == (3, 3, 0) and m["bets"] == 3
    settled = result.settled.set_index("home_name")
    assert settled["units"].to_dict() == pytest.approx(
        {"Wolverhampton Wanderers": 1.50, "Nottingham Forest": -1.0, "Dundee": 0.95})
    assert settled["status"].to_dict() == {"Wolverhampton Wanderers": "won", "Nottingham Forest": "lost", "Dundee": "won"}
    assert m["flat_stake_roi"] == pytest.approx((1.50 - 1.0 + 0.95) / 3) and m["hit_rate"] == pytest.approx(2 / 3)
    assert "flat_roi_ci_low" in m and m["flat_roi_ci_low"] <= m["flat_stake_roi"] <= m["flat_roi_ci_high"]
    # Closing-line value from Pinnacle's price an hour before kickoff, which moved towards every bet.
    assert (settled["closing_at"] == T2).all()
    assert settled["closing_lead_hours"].round(2).to_dict() == {
        "Wolverhampton Wanderers": 1.0, "Nottingham Forest": 3.0, "Dundee": 1.0}
    assert (settled["clv"] > settled["ev"]).all() and m["mean_clv"] > m["mean_ev"]
    assert settled.loc["Dundee", "clv"] == pytest.approx(_expected_ev((1.78, 3.90, 4.40), "home", 1.95))
    assert m["share_with_closing_move"] == 1.0 and m["mean_closing_lead_hours"] == pytest.approx(5 / 3)
    assert set(result.by_division["division"]) == {"E0", "SC0"} and set(result.by_bookmaker["bookmaker"]) == {
        "william_hill", "betvictor"}
    assert result.by_selection.set_index("selection").loc["away", "bets"] == 2
    assert result.by_source["source"].tolist() == ["odds_api"]
