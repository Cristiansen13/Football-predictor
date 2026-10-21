import json

import pandas as pd
import pytest

from football_predictor.etl import fbref, football_data, sofifa, understat

from . import fake_sources as fake


def test_football_data_results_and_odds():
    matches, odds = football_data.parse_season(fake.football_data_csv(), 2023)
    assert len(matches) == 10  # blank trailing row dropped
    first = matches.iloc[0]
    assert first["kickoff"] == pd.Timestamp("2023-08-12 15:00")
    assert (first["home_team"], first["away_team"]) == ("Man United", "Man City")
    assert (first["home_goals"], first["away_goals"]) == (1, 2)
    assert first["home_shots"] == 12 and first["away_shots_on_target"] == 3

    assert set(odds["bookmaker"]) == {"pinnacle", "average", "bet365"}
    closing = odds[(odds["bookmaker"] == "pinnacle") & (odds["market_time"] == "closing")].iloc[0]
    assert (closing["home"], closing["draw"], closing["away"]) == (2.08, 3.55, 3.90)
    assert len(odds) == 10 * 6


def test_football_data_two_digit_years_without_time():
    csv = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nE0,19/08/17,Watford,Liverpool,3,3\n"
    matches, odds = football_data.parse_season(csv, 2017)
    assert matches.loc[0, "kickoff"] == pd.Timestamp("2017-08-19")
    assert odds.empty


def test_understat_js_string_decoding():
    payload = {"name": "Martin Ødegaard", "quote": "it's", "nested": [1, 2.5, None]}
    escaped = fake.hex_escape(json.dumps(payload))
    html = f"<script>var playersData = JSON.parse('{escaped}');</script>"
    assert understat.extract_json_var(html, "playersData") == payload
    # A literal JS escape of the single quote also decodes.
    assert understat.extract_json_var("var x = JSON.parse('[\"it\\'s\"]')", "x") == ["it's"]
    with pytest.raises(ValueError):
        understat.extract_json_var(html, "datesData")


def test_understat_league_parsing():
    html = fake.understat_league_html()
    matches = understat.parse_league_matches(understat.extract_json_var(html, "datesData"), 2023)
    assert len(matches) == 12
    assert matches["is_result"].sum() == 10
    assert matches["home_goals"].isna().sum() == 2
    assert matches.loc[0, "home_xg"] == pytest.approx(1.3)

    history = understat.parse_team_history(understat.extract_json_var(html, "teamsData"))
    linked = understat.link_team_history(matches, history)
    assert len(linked) == 20  # two rows per played match
    home_row = linked[(linked["team"] == "Manchester United") & (linked["venue"] == "home")].iloc[0]
    assert home_row["ppda"] == pytest.approx(10.0)
    assert home_row["away_team"] == "Manchester City"


def test_understat_rosters():
    rosters = understat.extract_json_var(fake.understat_match_html("1"), "rostersData")
    frame = understat.parse_match_rosters(rosters, "1")
    assert frame["is_starter"].sum() == 4
    assert "André Onana" in set(frame["player_name"])
    assert frame.groupby("venue")["xa"].sum()["home"] == pytest.approx(0.45)


def test_fbref_schedule():
    schedule = fbref.parse_schedule(fake.fbref_schedule_html(), 2023)
    assert len(schedule) == 12  # spacer and repeated header rows skipped
    first = schedule.iloc[0]
    assert first["kickoff"] == pd.Timestamp("2023-08-12 15:00")
    assert (first["home_goals"], first["away_goals"]) == (1, 2)
    assert first["home_squad_id"] == "19538871" and first["home_squad_slug"] == "Manchester-United"
    assert schedule["home_goals"].isna().sum() == 2
    squads = fbref.squads_from_schedule(schedule)
    assert len(squads) == 4


def test_fbref_team_logs_including_commented_tables():
    schedule_log = fbref.parse_team_matchlogs(fake.fbref_team_log_html("Manchester City", "schedule"), "Manchester City")
    shooting_log = fbref.parse_team_matchlogs(fake.fbref_team_log_html("Manchester City", "shooting"), "Manchester City")
    assert "EFL Cup" in set(schedule_log["competition"])
    assert shooting_log["shots"].notna().all()
    league = fbref.combine_team_logs(schedule_log, shooting_log)
    assert len(league) == 5
    assert league["shots"].notna().all() and league["possession"].notna().all()


def test_sofifa_players_and_versions():
    html = fake.sofifa_players_html([("192985", "Kevin De Bruyne", "Manchester City", 91), ("209331", "Mohamed Salah", "Liverpool", 89)])
    players = sofifa.parse_players_page(html)
    assert list(players["sofifa_id"]) == ["192985", "209331"]
    assert players.loc[0, "player_name"] == "Kevin De Bruyne"  # full name from the tooltip
    assert players.loc[0, "overall"] == 91 and players.loc[0, "team"] == "Manchester City"

    versions = sofifa.parse_roster_versions(html)
    assert list(versions["roster_id"]) == ["240001", "240010"]
    assert versions.loc[1, "snapshot_date"] == pd.Timestamp("2023-10-06")


def test_sofifa_snapshot_selection():
    versions = pd.DataFrame(
        {"roster_id": [str(i) for i in range(10)], "snapshot_date": pd.date_range("2023-06-01", periods=10, freq="10D")}
    )
    chosen = sofifa.select_snapshots(versions, pd.Timestamp("2023-07-01"), pd.Timestamp("2023-09-01"), every_days=20)
    assert chosen["snapshot_date"].iloc[0] < pd.Timestamp("2023-07-01")
    assert chosen["snapshot_date"].diff().dropna().min() >= pd.Timedelta(days=10)


def test_football_data_fixtures_and_bom():
    matches, odds = football_data.parse_fixtures(fake.fixtures_csv())
    assert list(matches["home_team"]) == ["Wolves", "Nott'm Forest"]  # Scottish row filtered out
    assert matches["home_goals"].isna().all() and (matches["season"] == 2023).all()
    assert set(odds["market_time"]) == {"opening"} and set(odds["bookmaker"]) == {"pinnacle", "average", "bet365"}

    championship, odds = football_data.parse_season(fake.championship_csv(), 2023, "E1")
    assert len(championship) == 12 and (championship["division"] == "E1").all()  # BOM-prefixed header parsed
    assert set(odds["bookmaker"]) == {"average"}


def test_sofifa_editions_cover_seasons():
    assert sofifa.editions_for_seasons(2023, 2024) == [24, 25]


def test_espn_scoreboard_and_cup_schedule():
    from football_predictor.etl import espn

    day = pd.Timestamp("2023-08-12")
    board = espn.parse_scoreboard(fake.espn_scoreboard_json(day))
    assert len(board) == 4 and set(board["venue"]) == {"home", "away"}
    assert board["is_final"].all() and board["possession"].notna().all()
    first = board.iloc[0]
    assert first["kickoff"] == pd.Timestamp("2023-08-12 15:00")  # back in UK local time
    assert first["home_team"] == "Manchester United" and first["espn_team_id"] == "360"
    assert set(board.columns) >= {"shots", "shots_on_target", "goals"}

    cups = espn.parse_team_schedule(fake.espn_cup_schedule_json("Manchester City"), "Manchester City", "FA Cup")
    assert len(cups) == 1 and cups.iloc[0]["venue"] == "home" and cups.iloc[0]["competition"] == "FA Cup"
    assert cups.iloc[0]["kickoff"] == pd.Timestamp("2023-08-15 19:45")

    unplayed = espn.parse_scoreboard(fake.espn_scoreboard_json(pd.Timestamp("2023-09-16")))
    assert not unplayed["is_final"].any() and unplayed["goals"].isna().all()
