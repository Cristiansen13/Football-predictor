"""Offline stand-ins for the four data sources, built in each site's own format.

One mini league of four clubs is rendered as a Football-Data CSV, Understat
pages (hex-escaped JSON in <script> tags), FBref HTML tables (one hidden in a
comment) and a SOFIFA players page. Every source spells the clubs its own way.
"""

from __future__ import annotations

import json

import pandas as pd

from football_predictor.config import season_label
from football_predictor.etl import espn, fbref, understat

SEASON = 2023

# canonical -> per-source spelling
CLUBS = {
    "Manchester United": {"fd": "Man United", "us": "Manchester United", "fb": "Manchester Utd", "sf": "Manchester United"},
    "Manchester City": {"fd": "Man City", "us": "Manchester City", "fb": "Manchester City", "sf": "Manchester City"},
    "Nottingham Forest": {"fd": "Nott'm Forest", "us": "Nottingham Forest", "fb": "Nott'ham Forest", "sf": "Nottingham Forest"},
    "Wolverhampton Wanderers": {"fd": "Wolves", "us": "Wolverhampton Wanderers", "fb": "Wolves", "sf": "Wolverhampton Wanderers"},
}
SQUADS = {
    "Manchester United": ("19538871", "Manchester-United"),
    "Manchester City": ("b8fd03ef", "Manchester-City"),
    "Nottingham Forest": ("e4a775cb", "Nottingham-Forest"),
    "Wolverhampton Wanderers": ("8cec06e1", "Wolverhampton-Wanderers"),
}
TEAMS = list(CLUBS)


def fixtures() -> pd.DataFrame:
    """Double round robin, one round per Saturday; the last round is still to be played."""
    pairs = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]
    pairs += [(b, a) for a, b in pairs]
    scores = [(1, 2), (0, 0), (2, 1), (3, 0), (1, 1), (0, 1), (2, 2), (1, 0), (0, 2), (4, 1), (None, None), (None, None)]
    rows = []
    for k, ((h, a), (hg, ag)) in enumerate(zip(pairs, scores)):
        rows.append(
            {
                "kickoff": pd.Timestamp("2023-08-12 15:00") + pd.Timedelta(days=7 * (k // 2), hours=2 * (k % 2)),
                "home": TEAMS[h],
                "away": TEAMS[a],
                "home_goals": hg,
                "away_goals": ag,
                "home_xg": None if hg is None else round(0.4 + 0.9 * hg, 2),
                "away_xg": None if ag is None else round(0.3 + 0.8 * ag, 2),
                "possession": 40 + 3 * k,
            }
        )
    return pd.DataFrame(rows)


# -- Football-Data.co.uk ------------------------------------------------------


def football_data_csv() -> str:
    header = ("Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST,B365H,B365D,B365A,"
              "PSH,PSD,PSA,AvgH,AvgD,AvgA,B365CH,B365CD,B365CA,PSCH,PSCD,PSCA,AvgCH,AvgCD,AvgCA,"
              "B365>2.5,B365<2.5,P>2.5,P<2.5,PC>2.5,PC<2.5,AvgC>2.5,AvgC<2.5")
    lines = [header]
    for row in fixtures().itertuples():
        if row.home_goals is None or pd.isna(row.home_goals):
            continue
        ftr = "H" if row.home_goals > row.away_goals else "D" if row.home_goals == row.away_goals else "A"
        odds = ("2.10,3.40,3.60,2.15,3.45,3.70,2.05,3.35,3.55,2.00,3.50,3.80,2.08,3.55,3.90,2.02,3.45,3.70,"
                "1.90,1.95,1.92,1.98,1.88,2.02,1.86,2.00")
        lines.append(
            f"E0,{row.kickoff:%d/%m/%Y},{row.kickoff:%H:%M},{CLUBS[row.home]['fd']},{CLUBS[row.away]['fd']},"
            f"{int(row.home_goals)},{int(row.away_goals)},{ftr},12,9,5,3,{odds}"
        )
    return "\n".join(lines) + "\n,,,,,,,,,,,,,,,,,,,,,,,,,,,,,\n"  # trailing blank row like the real files


# Championship: two known clubs under Football-Data spellings, one abbreviated
# ("Peterboro") and one club that is in no register yet.
CHAMPIONSHIP = ["Leeds", "Sheffield Weds", "Peterboro", "Newtown Rangers"]
CHAMPIONSHIP_CANONICAL = ["Leeds United", "Sheffield Wednesday", "Peterborough United", "Newtown Rangers"]


def championship_csv() -> str:
    """Double round robin with two matches per round and no club twice in a round."""
    lines = ["\ufeffDiv,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,AvgH,AvgD,AvgA"]
    day = pd.Timestamp("2023-08-05 15:00")
    rounds = [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]]
    rounds += [[(a, h) for h, a in pairs] for pairs in rounds]
    for k, (h, a) in enumerate(pair for pairs in rounds for pair in pairs):
        kickoff = day + pd.Timedelta(days=7 * (k // 2))
        lines.append(f"E1,{kickoff:%d/%m/%Y},{kickoff:%H:%M},{CHAMPIONSHIP[h]},{CHAMPIONSHIP[a]},"
                     f"{k % 3},{(k + 1) % 2},H,2.4,3.3,3.0")
    return "\n".join(lines) + "\n"


def fixtures_csv() -> str:
    """Football-Data's weekly fixtures list: the two unplayed matches, plus a Scottish one."""
    return (
        "Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A,PSH,PSD,PSA,AvgH,AvgD,AvgA,Avg>2.5,Avg<2.5\n"
        "E0,16/09/2023,15:00,Wolves,Man United,3.10,3.50,2.30,3.20,3.55,2.35,3.05,3.45,2.28,2.05,1.80\n"
        "E0,16/09/2023,17:00,Nott'm Forest,Man City,9.00,5.50,1.33,9.50,5.60,1.35,8.80,5.40,1.32,1.60,2.35\n"
        "SC0,16/09/2023,15:00,Celtic,Rangers,1.80,3.80,4.20,1.82,3.85,4.30,1.79,3.75,4.10,1.70,2.15\n"
    )


# -- Understat -----------------------------------------------------------------


def hex_escape(text: str) -> str:
    """Understat-style encoding: every non-alphanumeric character as \\xHH."""
    return "".join(ch if ch.isalnum() and ch.isascii() else f"\\x{ord(ch):02X}" for ch in text)


def understat_payload() -> tuple[list, dict]:
    dates, teams = [], {}
    for k, row in enumerate(fixtures().itertuples()):
        played = row.home_goals is not None and not pd.isna(row.home_goals)
        dates.append(
            {
                "id": str(22000 + k),
                "isResult": played,
                "h": {"id": str(TEAMS.index(row.home)), "title": CLUBS[row.home]["us"], "short_title": "H"},
                "a": {"id": str(TEAMS.index(row.away)), "title": CLUBS[row.away]["us"], "short_title": "A"},
                "goals": {"h": str(int(row.home_goals)) if played else None, "a": str(int(row.away_goals)) if played else None},
                "xG": {"h": str(row.home_xg) if played else None, "a": str(row.away_xg) if played else None},
                "datetime": f"{row.kickoff:%Y-%m-%d %H:%M:%S}",
            }
        )
        if not played:
            continue
        for side, team, xg, xga in (("h", row.home, row.home_xg, row.away_xg), ("a", row.away, row.away_xg, row.home_xg)):
            entry = teams.setdefault(str(TEAMS.index(team)), {"id": str(TEAMS.index(team)), "title": CLUBS[team]["us"], "history": []})
            entry["history"].append(
                {
                    "h_a": side, "xG": xg, "xGA": xga, "npxG": xg, "npxGA": xga,
                    "ppda": {"att": 300, "def": 30 if side == "h" else 25},
                    "ppda_allowed": {"att": 280, "def": 20},
                    "deep": 7, "deep_allowed": 4, "xpts": 1.4,
                    "date": f"{row.kickoff:%Y-%m-%d %H:%M:%S}",
                }
            )
    return dates, teams


def understat_league_html() -> str:
    dates, teams = understat_payload()
    return (
        "<html><body><script>\n"
        f"\tvar datesData\t= JSON.parse('{hex_escape(json.dumps(dates))}');\n"
        f"\tvar teamsData = JSON.parse('{hex_escape(json.dumps(teams))}');\n"
        "</script></body></html>"
    )


def understat_match_html(understat_id: str) -> str:
    starters = {"h": ["André Onana", "Bruno Fernandes"], "a": ["Ederson", "Kevin De Bruyne"]}
    rosters = {
        side: {
            str(i): {"player": name, "position": "GK" if i == 0 else "AMC", "time": "90", "xG": "0.1", "xA": "0.2"}
            for i, name in enumerate(names)
        }
        for side, names in starters.items()
    }
    rosters["h"]["9"] = {"player": "Sub Player", "position": "Sub", "time": "10", "xG": "0", "xA": "0.05"}
    return f"<script>var rostersData = JSON.parse('{hex_escape(json.dumps(rosters))}');</script>"


# -- FBref ---------------------------------------------------------------------


def _squad_link(team: str) -> str:
    squad_id, slug = SQUADS[team]
    return f'<a href="/en/squads/{squad_id}/{slug}-Stats">{CLUBS[team]["fb"]}</a>'


def fbref_schedule_html() -> str:
    rows = []
    for k, row in enumerate(fixtures().itertuples()):
        played = row.home_goals is not None and not pd.isna(row.home_goals)
        score = f'<a href="/en/matches/{k}">{int(row.home_goals)}–{int(row.away_goals)}</a>' if played else ""
        rows.append(
            f'<tr><th data-stat="gameweek">{k // 2 + 1}</th>'
            f'<td data-stat="dayofweek">Sat</td>'
            f'<td data-stat="date"><a href="/en/matches/{row.kickoff:%Y-%m-%d}">{row.kickoff:%Y-%m-%d}</a></td>'
            f'<td data-stat="start_time"><span class="venuetime">{row.kickoff:%H:%M}</span> '
            f'<span class="localtime">(16:00)</span></td>'
            f'<td data-stat="home_team">{_squad_link(row.home)}</td>'
            f'<td data-stat="home_xg">{row.home_xg if played else ""}</td>'
            f'<td data-stat="score">{score}</td>'
            f'<td data-stat="away_xg">{row.away_xg if played else ""}</td>'
            f'<td data-stat="away_team">{_squad_link(row.away)}</td></tr>'
        )
        if k == 5:
            rows.append('<tr class="spacer"><td colspan="9"></td></tr>')
            rows.append('<tr class="thead"><th data-stat="gameweek">Wk</th><td data-stat="date">Date</td></tr>')
    return (
        f'<html><body><table class="stats_table" id="sched_{season_label(SEASON)}_9_1">'
        "<thead><tr><th>Wk</th><th>Day</th><th>Date</th><th>Time</th><th>Home</th><th>xG</th>"
        "<th>Score</th><th>xG</th><th>Away</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></body></html>"
    )


def team_log_rows(team: str) -> list[dict]:
    rows = []
    for row in fixtures().itertuples():
        if team not in (row.home, row.away) or row.home_goals is None or pd.isna(row.home_goals):
            continue
        home = team == row.home
        rows.append(
            {
                "kickoff": row.kickoff,
                "comp": "Premier League",
                "venue": "Home" if home else "Away",
                "opponent": CLUBS[row.away if home else row.home]["fb"],
                "goals_for": row.home_goals if home else row.away_goals,
                "goals_against": row.away_goals if home else row.home_goals,
                "possession": row.possession if home else 100 - row.possession,
                "shots": 14 if home else 8,
            }
        )
    if team == "Manchester City":  # a midweek cup tie, away, three days before matchweek 3
        rows.append({"kickoff": pd.Timestamp("2023-08-23 19:45"), "comp": "EFL Cup", "venue": "Away",
                     "opponent": "Newcastle Utd", "goals_for": 1, "goals_against": 0, "possession": 60, "shots": 11})
    return sorted(rows, key=lambda r: r["kickoff"])


def fbref_team_log_html(team: str, kind: str) -> str:
    body = []
    for r in team_log_rows(team):
        if kind == "shooting" and r["comp"] != "Premier League":
            continue
        cells = (
            f'<th data-stat="date"><a href="/en/matches/x">{r["kickoff"]:%Y-%m-%d}</a></th>'
            f'<td data-stat="start_time">{r["kickoff"]:%H:%M}</td>'
            f'<td data-stat="comp"><a href="/en/comps/x">{r["comp"]}</a></td>'
            f'<td data-stat="venue">{r["venue"]}</td>'
            f'<td data-stat="goals_for">{r["goals_for"]}</td><td data-stat="goals_against">{r["goals_against"]}</td>'
            f'<td data-stat="opponent"><a href="/en/squads/x">{r["opponent"]}</a></td>'
        )
        if kind == "schedule":
            cells += f'<td data-stat="possession">{r["possession"]}</td>'
        else:
            cells += f'<td data-stat="shots">{r["shots"]}</td><td data-stat="shots_on_target">{r["shots"] // 2}</td>'
        body.append(f"<tr>{cells}</tr>")
    table = f'<table id="matchlogs_for"><tbody>{"".join(body)}</tbody></table>'
    # Shooting logs ship inside an HTML comment, as on FBref.
    return f"<html><body><div>{'<!--' + table + '-->' if kind == 'shooting' else table}</div></body></html>"


# -- SOFIFA --------------------------------------------------------------------


def sofifa_players_html(players: list[tuple[str, str, str, int]]) -> str:
    rows = "".join(
        f'<tr><td class="col-name"><a href="/player/{pid}/x/240001/" data-tippy-content="{name}">{name.split()[-1]}</a>'
        f'<span class="pos">ST</span></td><td data-col="oa"><em>{ovr}</em></td>'
        f'<td class="col-name"><a href="/team/{abs(hash(team)) % 1000}/x/">{team}</a></td></tr>'
        for pid, name, team, ovr in players
    )
    versions = '<select name="r"><option value="240001">Sep 1, 2023</option><option value="240010">Oct 6, 2023</option></select>'
    return f"<html><body>{versions}<table><tbody>{rows}</tbody></table></body></html>"


# -- HTTP stub -----------------------------------------------------------------


# -- ESPN ----------------------------------------------------------------------

ESPN_IDS = {"Manchester United": "360", "Manchester City": "382", "Nottingham Forest": "393",
            "Wolverhampton Wanderers": "380"}


def _espn_event(event_id: str, kickoff: pd.Timestamp, home: str, away: str, home_goals, away_goals,
                stats: dict[str, dict] | None = None) -> dict:
    final = home_goals is not None
    utc = kickoff.tz_localize("Europe/London").tz_convert("UTC")
    competitors = []
    for side, team, goals in (("home", home, home_goals), ("away", away, away_goals)):
        competitor = {"homeAway": side, "team": {"id": ESPN_IDS[team], "displayName": CLUBS[team]["us"]},
                      "score": "" if goals is None else str(goals)}
        if stats and team in stats:
            competitor["statistics"] = [{"name": name, "displayValue": str(value)}
                                        for name, value in stats[team].items()]
        competitors.append(competitor)
    return {"id": event_id, "date": utc.strftime("%Y-%m-%dT%H:%MZ"),
            "competitions": [{"competitors": competitors,
                              "status": {"type": {"name": "STATUS_FULL_TIME" if final else "STATUS_SCHEDULED"}}}]}


def espn_scoreboard_json(day: pd.Timestamp) -> str:
    """Premier League scoreboard for one day with possession and shots per side."""
    events = []
    for k, row in fixtures().iterrows():
        if row["kickoff"].normalize() != day.normalize():
            continue
        played = pd.notna(row["home_goals"])  # the frame holds NaN for the unplayed round
        stats = None
        if played:
            stats = {row["home"]: {"possessionPct": row["possession"], "totalShots": 10 + k, "shotsOnTarget": 4},
                     row["away"]: {"possessionPct": 100 - row["possession"], "totalShots": 8 + k, "shotsOnTarget": 3}}
        goals = (int(row["home_goals"]), int(row["away_goals"])) if played else (None, None)
        events.append(_espn_event(f"7{k:05d}", row["kickoff"], row["home"], row["away"], *goals, stats))
    return json.dumps({"events": events})


def espn_cup_schedule_json(team: str) -> str:
    """One FA Cup tie per club, on the Tuesday before the second league round."""
    opponent = next(t for t in TEAMS if t != team)
    event = _espn_event("8" + ESPN_IDS[team], pd.Timestamp("2023-08-15 19:45"), team, opponent, 2, 1)
    return json.dumps({"team": {"id": ESPN_IDS[team]}, "events": [event]})


class FakeClient:
    def __init__(self, *, understat_api: bool = False) -> None:
        """``understat_api``: serve the league data only through Understat's JSON
        endpoint, as when the page stops embedding it."""
        label = season_label(SEASON)
        self.pages = {
            "https://www.football-data.co.uk/mmz4281/2324/E0.csv": football_data_csv(),
            "https://www.football-data.co.uk/mmz4281/2324/E1.csv": championship_csv(),
            "https://www.football-data.co.uk/fixtures.csv": fixtures_csv(),
            understat.LEAGUE_URL.format(season=SEASON): understat_league_html(),
            fbref.SCHEDULE_URL.format(label=label): fbref_schedule_html(),
        }
        if understat_api:
            dates, teams = understat_payload()
            self.pages[understat.LEAGUE_URL.format(season=SEASON)] = "<html><body>loading...</body></html>"
            self.pages[understat.LEAGUE_API_URL.format(season=SEASON)] = json.dumps(
                {"dates": dates, "teams": teams, "players": []}
            )
        for k in range(12):
            self.pages[understat.MATCH_URL.format(match_id=22000 + k)] = understat_match_html(str(22000 + k))
        for team, (squad_id, slug) in SQUADS.items():
            self.pages[fbref.TEAM_SCHEDULE_URL.format(squad_id=squad_id, label=label, slug=slug)] = (
                fbref_team_log_html(team, "schedule")
            )
            self.pages[fbref.TEAM_SHOOTING_URL.format(squad_id=squad_id, label=label, slug=slug)] = (
                fbref_team_log_html(team, "shooting")
            )
        for day in fixtures()["kickoff"].dt.normalize().unique():
            self.pages[espn.SCOREBOARD_URL.format(league="eng.1", date=pd.Timestamp(day))] = espn_scoreboard_json(
                pd.Timestamp(day)
            )
        for team, espn_id in ESPN_IDS.items():
            for cup in espn.CUPS:
                payload = espn_cup_schedule_json(team) if cup == "eng.fa" else json.dumps(
                    {"team": {"id": espn_id}, "events": []}
                )
                self.pages[espn.TEAM_SCHEDULE_URL.format(league=cup, team_id=espn_id, season=SEASON)] = payload
        self.requested: list[str] = []

    def get_text(self, url: str, **kwargs) -> str:
        self.requested.append((url, kwargs.get("headers")) if kwargs.get("headers") else url)
        if url not in self.pages:
            raise AssertionError(f"Unexpected request: {url}")
        return self.pages[url]
