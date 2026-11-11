# Data

Everything here is produced by the scripts in [`extraction/`](../extraction):

```bash
python -m extraction.download --seasons 2014-2025   # data/raw
python -m extraction.preprocess                      # data/final/matches.csv
python -m extraction.features                        # data/final/features.csv
```

## `raw/`: the files as downloaded

One CSV per Premier League season from
[Football-Data.co.uk](https://www.football-data.co.uk/englandm.php), saved
untouched: `E0_2014.csv` (season 2014/15) to `E0_2025.csv` (season 2025/26).
The address of a file is `https://www.football-data.co.uk/mmz4281/<season>/E0.csv`,
where `<season>` is `1415` for 2014/15 and `E0` is the site's code for the
Premier League.

| | |
| :--- | :--- |
| Files | 12, 1.8 MB in total |
| Rows | 380 per file, one per match |
| Columns | 62 to 132 depending on the season: date, teams, score, referee, match statistics, and the odds of up to ten bookmakers |
| Column notes | <https://www.football-data.co.uk/notes.txt> |

## `final/`: the datasets the analysis uses

### `matches.csv`: 4,560 matches, 31 columns

One row per match, cleaned and renamed.

| Columns | Meaning |
| :--- | :--- |
| `match_id`, `season`, `date`, `home`, `away`, `referee` | Identity. A season is named by the year it starts in (2023 = 2023/24). |
| `home_goals`, `away_goals`, `result` | Full-time score and result: `H` home win, `D` draw, `A` away win. |
| `home_goals_ht`, `away_goals_ht` | Half-time score. |
| `home_shots`, `away_shots`, `home_shots_target`, `away_shots_target` | Shots and shots on target. |
| `home_corners`, `away_corners`, `home_fouls`, `away_fouls` | Corners and fouls committed. |
| `home_yellow`, `away_yellow`, `home_red`, `away_red` | Cards. |
| `odds_home`, `odds_draw`, `odds_away` | Bet365 decimal odds for the three results, collected days before the match. |
| `closing_home`, `closing_draw`, `closing_away` | Bet365 odds at kickoff. Published from 2019/20; empty for the 1,900 earlier matches. |
| `odds_over`, `odds_under` | Bet365 odds for more or fewer than 2.5 goals. Published from 2019/20. |

### `features.csv`: 4,560 matches, 21 features

The modelling table: what was known before kickoff, and the result to predict.
Every feature is computed only from matches played earlier.

| Columns | Meaning |
| :--- | :--- |
| `match_id`, `season`, `date`, `home`, `away` | Identity, the same as in `matches.csv`. |
| `home_elo`, `away_elo`, `elo_diff` | Elo rating of each side and their difference. |
| `home_pts5`, `away_pts5` | Points per game over the last 5 league matches. |
| `home_gf5`, `home_ga5`, `away_gf5`, `away_ga5` | Goals scored and conceded per game over the last 5. |
| `home_sot5`, `home_sota5`, `away_sot5`, `away_sota5` | Shots on target for and against per game over the last 5. |
| `home_gf38`, `home_ga38`, `away_gf38`, `away_ga38` | Goals scored and conceded per game over the last 38 (one season). |
| `home_venue_pts5`, `away_venue_pts5` | Points per game over the last 5 matches at this venue. |
| `home_rest`, `away_rest` | Days since the previous league match, capped at 14. |
| `result`, `outcome` | The target: `H`/`D`/`A`, and the same as 0/1/2. |

A feature is empty where there is not enough history yet (a club's first
matches in the data): 247 rows have at least one empty feature.
