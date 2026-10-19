"""Offline stand-ins for the four data sources, built in each site's own format.

One mini league of four clubs is rendered as a Football-Data CSV, Understat
pages (hex-escaped JSON in <script> tags), FBref HTML tables (one hidden in a
comment) and a SOFIFA players page. Every source spells the clubs its own way.
"""

from __future__ import annotations

import json

import pandas as pd

from football_predictor.config import season_label
from football_predictor.etl import fbref, understat

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
              "PSH,PSD,PSA,AvgH,AvgD,AvgA,B365CH,B365CD,B365CA,PSCH,PSCD,PSCA,AvgCH,AvgCD,AvgCA")
    lines = [header]
    for row in fixtures().itertuples():
        if row.home_goals is None or pd.isna(row.home_goals):
            continue
        ftr = "H" if row.home_goals > row.away_goals else "D" if row.home_goals == row.away_goals else "A"
        odds = "2.10,3.40,3.60,2.15,3.45,3.70,2.05,3.35,3.55,2.00,3.50,3.80,2.08,3.55,3.90,2.02,3.45,3.70"
        lines.append(
            f"E0,{row.kickoff:%d/%m/%Y},{row.kickoff:%H:%M},{CLUBS[row.home]['fd']},{CLUBS[row.away]['fd']},"
            f"{int(row.home_goals)},{int(row.away_goals)},{ftr},12,9,5,3,{odds}"
        )
    return "\n".join(lines) + "\n,,,,,,,,,,,,,,,,,,,,,,,,,,,,,\n"  # trailing blank row like the real files


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


class FakeClient:
    def __init__(self) -> None:
        label = season_label(SEASON)
        self.pages = {
            f"https://www.football-data.co.uk/mmz4281/2324/E0.csv": football_data_csv(),
            understat.LEAGUE_URL.format(season=SEASON): understat_league_html(),
            fbref.SCHEDULE_URL.format(label=label): fbref_schedule_html(),
        }
        for k in range(12):
            self.pages[understat.MATCH_URL.format(match_id=22000 + k)] = understat_match_html(str(22000 + k))
        for team, (squad_id, slug) in SQUADS.items():
            self.pages[fbref.TEAM_SCHEDULE_URL.format(squad_id=squad_id, label=label, slug=slug)] = (
                fbref_team_log_html(team, "schedule")
            )
            self.pages[fbref.TEAM_SHOOTING_URL.format(squad_id=squad_id, label=label, slug=slug)] = (
                fbref_team_log_html(team, "shooting")
            )
        self.requested: list[str] = []

    def get_text(self, url: str, **kwargs) -> str:
        self.requested.append(url)
        if url not in self.pages:
            raise AssertionError(f"Unexpected request: {url}")
        return self.pages[url]
