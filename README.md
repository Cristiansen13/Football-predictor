# Football Predictor

Premier League match-outcome model built on a Dixon-Coles bivariate Poisson
framework, following [`project_architecture.md`](project_architecture.md):

| Architecture section | Implementation |
| :--- | :--- |
| 1. ETL & entity resolution | `etl/` scrapers, `etl/entity_resolution.py`, SQLite warehouse in `db.py` |
| 2. Temporal features | `features/` (time-decayed EWMA, fatigue index, lineup strength) |
| 3. Dixon-Coles engine | `models/dixon_coles.py` (time-weighted MLE with ρ and ξ) |
| 4. Probability matrix | `models/probability.py` (0-0 … 6-6 grid → 1X2, O/U 2.5, BTTS) |
| 5. Validation & backtesting | `evaluation/` (Brier, RPS, log loss, EV, fractional Kelly, PnL curve) |

## Install

```bash
pip install -e ".[dev]"            # core + pytest
pip install -e ".[scraping]"       # Playwright, only needed for SOFIFA
```

## Quick start

No network needed: run a walk-forward backtest on a simulated league.

```bash
football-predictor demo --features form --tune-xi
```

With real data:

```bash
# 1. ETL into data/football.db (season = start year; 2016-2024 is 2016/17 to 2024/25)
football-predictor ingest --seasons 2016-2024                                  # Football-Data + Understat
football-predictor ingest --seasons 2016-2024 --sources fbref                  # possession, shots, cup fixtures
football-predictor ingest --seasons 2016-2024 --sources understat --lineups    # starting XIs + xA (slow)
football-predictor ingest --seasons 2016-2024 --sources sofifa --chromium-path /path/to/chrome

# 2. Model
football-predictor ratings --as-of 2025-05-01
football-predictor predict "Man Utd" "Spurs" --as-of 2025-05-04
football-predictor predict --upcoming 7

# 3. Backtest the final season: xi picked on the previous season, EV bets at closing odds
football-predictor backtest --test-season 2024 --tune-xi --features form --ev-threshold 0.05 --kelly 0.25
```

`backtest` writes `data/reports/pnl_<season>.png` (cumulative PnL curve) and CSVs
of every prediction and bet. Use `--data-dir` to change the output location.

## Pipeline

### 1. Data ingestion and entity resolution

| Source | Module | Method | Loaded into |
| :--- | :--- | :--- | :--- |
| Football-Data.co.uk | `etl/football_data.py` | direct CSV | `matches`, `odds` (opening and closing, per bookmaker), shot counts |
| Understat | `etl/understat.py` | regex over `JSON.parse('\x..')` script tags, AJAX endpoint fallback | `advanced_stats` (xG, npxG, PPDA, deep completions; xA from match rosters), `lineups` |
| FBref | `etl/fbref.py` | BeautifulSoup over `data-stat` cells, including tables hidden in HTML comments | `matches`, `match_stats` (possession, shots), `team_schedule` (all competitions) |
| SOFIFA | `etl/sofifa.py` | Playwright (headless Chromium) | `player_ratings`, one row per player per roster snapshot |

Every source name goes through `FuzzyMatcher`. It normalises the name (accents,
`&`, abbreviations such as *Utd*, *Man*, *Nott'm*), looks it up in a curated alias
table, and otherwise matches on Levenshtein similarity plus a token-prefix rule
(*Tottenham* → *Tottenham Hotspur*). Ambiguous names such as *Manchester* are
rejected rather than guessed. Each club gets a deterministic UUIDv5. A fixture's
`match_id` is derived from `(season, home_id, away_id)`, so each source computes
the same key on its own and the `v_match_dataset` view joins them with plain SQL.
The pipeline is idempotent. If one source fails (blocked, layout changed), it is
reported and the other sources still load.

### 2. Point-in-time features

All features use only information available before kickoff. A test changes
future results and checks that earlier features stay identical.

* **EWMA** (`features/ewma.py`): a calendar-time half-life (default 45 days),
  so a game four days ago outweighs one a month ago. Values are shrunk towards
  the league mean of earlier matches, which handles cold starts. It runs over
  goals, xG, npxG, xA, shots, possession, PPDA, etc.
* **Fatigue** (`features/fatigue.py`): `rest_days` measured to the previous
  fixture in any competition (FBref), `fatigue_index = 1 / rest_days`, and a
  penalty that rises linearly below 72 h of rest, ×1.5 after an away game.
* **Lineup strength** (`features/lineup.py`): `lineup_score` is the mean OVR
  of the starting XI from the last SOFIFA snapshot before match day.
  `lineup_fraction` is that score divided by the squad's best-XI score.
  Unmatched starters get the squad's lower-quartile rating.

### 3. Dixon-Coles model

```
log λ = c + γ + attack[home] + defence[away] + θ_att·A(home) + θ_def·D(away)
log μ = c     + attack[away] + defence[home] + θ_att·A(away) + θ_def·D(home)
```

* `defence` measures weakness (a higher value concedes more). Attack and
  defence each sum to zero.
* The likelihood includes the Dixon-Coles τ correction on 0-0, 1-0, 0-1 and 1-1,
  with ρ fitted inside bounds.
* Each match is weighted by `exp(-ξ·days)` (ξ = 0.0019/day by default, about a
  one-year half-life). `backtest --tune-xi` picks ξ by walk-forward Brier score
  on the season before the test season.
* Optional covariates are standardised features of the scoring side (A) and the
  conceding side (D). Presets: `none` (pure Dixon-Coles), `form` (EWMA xG), and
  `full` (adds fatigue and lineup fraction).
* The fit is L-BFGS-B with an analytic gradient (checked against finite
  differences in the tests), a small ridge penalty, and warm starts during
  walk-forward refits. A full refit takes well under a second.
* A team missing from the training data (usually newly promoted) is rated at
  the 20th percentile of attack and the mirrored percentile of defence.

### 4. Probability matrix

`score_matrices` builds the 7×7 joint grid. The probability mass beyond six goals
is renormalised away, so 1X2 probabilities sum to one. `market_probabilities`
aggregates it to home/draw/away, over/under 2.5 and both-teams-to-score.

### 5. Validation and backtesting

`run_backtest` refits every `--refit-days` using only matches before each block.
It reports Brier score, RPS and log loss for the model and for the de-margined
market. It then bets the highest-EV outcome whenever
`EV = P_model × odds − 1 > threshold`, staking fractional Kelly on the
start-of-day bankroll, with per-bet and per-day caps. The bookmaker used is the
first available of Pinnacle closing, market-average closing, Bet365 closing,
then opening prices.

## Caveats

* The scrapers follow each site's current markup. They are tested offline
  against fixtures that reproduce those formats (`tests/fake_sources.py`), not
  against the live sites. Expect to adjust selectors when a site changes.
* FBref allows about 10 requests a minute, so the client throttles per host and
  caches pages under `data/cache/`. Pages for the current season are never
  cached. Scraping lineups costs one request per match.
* SOFIFA needs a Chromium build that matches the installed Playwright version
  (`playwright install chromium`), or pass `--chromium-path`.

## Tests

```bash
pytest
```

The suite covers entity resolution, all four parsers, the full ETL → SQL → feature
path on fake sources, feature leakage, parameter recovery on simulated leagues,
the gradient check, the scoring rules, Kelly settlement and the walk-forward
backtest.
