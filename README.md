# Football Predictor

Premier League match-outcome model built on a Dixon-Coles bivariate Poisson
framework, following [`project_architecture.md`](project_architecture.md):

| Architecture section | Implementation |
| :--- | :--- |
| 1. ETL & entity resolution | `etl/` scrapers, `etl/entity_resolution.py`, SQLite warehouse in `db.py`, `quality.py` |
| 2. Temporal features | `features/` (time-decayed EWMA, fatigue index, lineup strength, point-in-time audit) |
| 3. Dixon-Coles engine | `models/dixon_coles.py` (time-weighted MLE with ρ, ξ, xG weighting, multiple divisions) |
| 4. Probability matrix | `models/probability.py` (0-0 … 6-6 grid → 1X2, O/U 2.5, BTTS) |
| 5. Validation & backtesting | `evaluation/` (Brier, RPS, log loss, calibration, bootstrap CIs, EV, fractional Kelly, PnL) |

## Install

```bash
make install           # pip install -e ".[dev,scraping]"
make install-browser   # Chromium for SOFIFA and for sites that block plain HTTP clients
```

## Quick start

No network needed: a full walk-forward backtest on a simulated two-division league.

```bash
make demo              # football-predictor demo --features form --tune --compare
```

With real data:

```bash
make check             # is every source reachable and parseable right now?
make data              # results + odds (PL and Championship), xG/PPDA, possession, cup fixtures (ESPN)
make data-fbref        # possession and cup fixtures from FBref instead (see the caveat on anti-bot challenges)
make data-lineups      # + starting XIs with each player's xG/xA from 2022/23 (about 25 min per season, resumable)
make data-full         # + SOFIFA ratings (behind Cloudflare at the moment)
make status            # coverage per season, data issues, leakage audit
make backtest          # tune ξ and xG weight on the two previous seasons, test on the last completed one
make backtest-rolling  # one walk-forward over 2021/22-2025/26: five seasons of scored matches, per-season table
make backtest-market   # price-shopping benchmark: the sharp closing line as fair value, bet the best price
make predict           # this week's fixtures: probabilities, EV against current prices
```

Each target is a thin wrapper around the CLI:

| Command | Purpose |
| :--- | :--- |
| `football-predictor ingest [--seasons 2014-2025] [--sources …\|all] [--lineups]` | ETL into `data/football.db`; exits non-zero if a source failed |
| `football-predictor check-sources [--season 2024]` | fetch and parse one live page per source |
| `football-predictor status [--audit N]` | data coverage report, quality issues, point-in-time audit of N matches |
| `football-predictor ratings [--as-of DATE]` | team attack/defence ratings |
| `football-predictor predict HOME AWAY` / `--upcoming DAYS` | score grid, markets, best-EV bet and stake |
| `football-predictor backtest [--test-season Y \| --test-seasons A-B] [--tune] [--compare] [--markets 1x2 totals] [--fair model\|market\|blend] [--price closing\|opening\|max]` | walk-forward evaluation and betting simulation, 1X2 and over/under 2.5 |

Reports go to `data/reports/` (`metrics_<season>.json`, predictions, bets,
calibration table, PnL curve and reliability diagram). Use `--data-dir` to move them.

## 1. Data ingestion and entity resolution

| Source | Module | Method | Loaded into |
| :--- | :--- | :--- | :--- |
| Football-Data.co.uk | `etl/football_data.py` | direct CSV, Premier League (`E0`) and Championship (`E1`), plus the weekly `fixtures.csv` | `matches`, `odds` (1X2, opening/closing per bookmaker), `totals_odds` (over/under 2.5), shot counts |
| Understat | `etl/understat.py` | regex over `JSON.parse('\x..')` script tags; falls back to its JSON endpoint | `advanced_stats` (xG, npxG, PPDA, deep completions; xA from match rosters), `lineups` |
| ESPN | `etl/espn.py` | public JSON endpoints of ESPN's own site (no key, not a documented API) | `match_stats` (possession, shots), `team_schedule` (league, cups, Europe) |
| FBref | `etl/fbref.py` | BeautifulSoup over `data-stat` cells, including tables hidden in HTML comments | `matches`, `match_stats` (possession, shots), `team_schedule` (all competitions) |
| SOFIFA | `etl/sofifa.py` | Playwright (headless Chromium), every game edition's roster updates | `player_ratings`, one row per player per snapshot |

* **Blocked requests.** FBref and Understat increasingly answer scripted
  clients with Cloudflare challenges. `HttpClient` detects a block status or a
  challenge page and re-renders that URL in a headless browser
  (`etl/browser.py`). Use `--headful` if the headless browser is challenged too.
  Pages are throttled per host (FBref allows about 10 requests a minute) and
  cached under `data/cache/`. The current season is never cached.
* **Entity resolution.** `FuzzyMatcher` normalises names (accents, `&`, *Utd*,
  *Man*, *Nott'm*, …), checks a curated alias table, then scores Levenshtein
  similarity plus a token-prefix rule (*Peterboro* → *Peterborough United*).
  Ambiguous names such as *Manchester* are rejected. A brand-new lower-division
  club is registered only when nothing scores above 0.75, so a near-miss
  spelling never creates a duplicate. Each club gets a deterministic UUIDv5,
  and a fixture's `match_id` is derived from `(season, home_id, away_id)`, so
  every source computes the same key and `v_match_dataset` joins them in plain SQL.
* **Robust loading.** Every (source, season) step is idempotent. A failing
  source is reported and the others still load. SOFIFA snapshots and Understat
  lineups already in the database are skipped, so long downloads resume.
  Databases from older versions are migrated in place.
* **Quality checks.** `status` reports, per division and season, the club
  count, results versus expected (380 / 552), and coverage of xG, PPDA,
  possession, shots, closing odds, lineups, cup schedules and rating snapshots.
  It flags anything the model depends on that is missing.

## 2. Point-in-time features

Every feature is computed from information available before kickoff.
`status --audit N` checks this on real data. For N random matches it rebuilds
the feature table from a copy truncated at the match's kickoff, with the
match's own statistics blanked, and requires every feature to come out the same.
A test shows the audit catches a deliberately leaky feature.

* **EWMA** (`features/ewma.py`): calendar-time half-life (default 45 days), so a
  game four days ago outweighs one a month ago. Values are shrunk towards the
  league mean of earlier matches. Rows that share a timestamp never see each
  other's results.
* **Fatigue** (`features/fatigue.py`): rest to the previous fixture in any
  competition, `fatigue_index = 1 / rest_days`, and a penalty below 72 h of rest
  (×1.5 after an away game). A fixture listed twice by different sources a few
  hours apart counts as one match, not as zero rest.
* **Lineup strength** (`features/lineup.py`): mean SOFIFA overall rating of the
  starting XI from the last snapshot before match day, and its ratio to the
  squad's best XI. Without SOFIFA ratings the same two numbers come from the
  starters' own form: each player's expected xG + xA per 90 over his earlier
  matches on Understat (any club, one-year half-life, shrunk towards the
  league rate over the first 600 minutes), summed over the eleven starters and
  compared with the club's best eleven of the past year.

## 3. Dixon-Coles model

```
log λ = c[div] + γ[div] + attack[home] + defence[away] + θ_att·A(home) + θ_def·D(away)
log μ = c[div]          + attack[away] + defence[home] + θ_att·A(away) + θ_def·D(home)
```

* **Dixon-Coles core.** The ρ correction applies to 0-0, 1-0, 0-1 and 1-1, and
  each match is weighted by `exp(−ξ·days)`. The fit is L-BFGS-B with an analytic
  gradient (checked against finite differences) and warm starts between refits.
* **Training divisions.** The model can rate Premier League and Championship
  clubs on one scale (`--train-divisions E0 E1` or `all`): clubs carry their
  ratings through promotion and relegation, each division keeps its own scoring
  level and home advantage, and the ridge penalty shrinks each club towards its
  own division's mean. On simulated leagues this halves the rating error for
  newly promoted clubs. On real data it does the opposite, so the default
  trains on the predicted division only: Burnley and Leeds dominated the
  2024/25 Championship and carried ratings into 2025/26 far above what the
  market gave them, while Sunderland was under-rated. Matches involving the
  three promoted clubs accounted for nearly the whole of the model's deficit
  to the market that season; with Premier League data only, a promoted club
  starts at the unseen-club quantile and is learnt from its top-flight matches.
* **xG in the likelihood** (`--xg-weight`, CLI default 0.3). Each match
  contributes `(1−a)·log L(goals) + a·log L(xG)`. Ratings settle faster because
  xG is far less noisy than goals. Matches without xG (the Championship) use
  goals only.
* **Optional covariates.** Standardised features of the scoring side (A) and
  the conceding side (D). Presets: `none`, `form` (EWMA xG), `lineup` (the
  starting XI's form and its fraction of the regular XI) and `full` (EWMA xG,
  fatigue and lineup fraction). `--attack-features` / `--defence-features`
  override them.
* **Tuning.** `backtest --tune` grid-searches ξ × xG weight by walk-forward
  Brier score on the seasons before the test season (`--tune-seasons`, default
  2, in `--jobs` parallel processes). One validation season is not enough: on
  real data the ranking of settings between 2023/24 and 2024/25 had a rank
  correlation of −0.8, and the setting chosen on one season alone was the worst
  of the grid on the next.

## 4. Probability matrix

`score_matrices` builds the 7×7 joint grid. Mass beyond six goals is
renormalised away, so the 1X2 probabilities sum to one. `market_probabilities`
aggregates the grid into home/draw/away, over/under 2.5 and both-teams-to-score.

## 5. Validation and backtesting

`run_backtest` refits every `--refit-days` using only matches before each block.
It trains on `--train-divisions` (default: the scored division) and scores only
the Premier League. Metrics:

* **Forecast quality:** Brier (primary), RPS, log loss, exact-score log loss,
  accuracy, expected calibration error, and Brier skill against the historical
  base rates.
* **Against the market:** the same scores for the de-margined closing prices,
  Brier skill against the market, and a paired-bootstrap 95% interval for
  "model Brier − market Brier". An edge is only credible if the interval stays
  below zero.
* **Over/under 2.5 and both teams to score:** binary Brier and log loss of
  `p_over_2_5` against the de-margined over/under prices (paired bootstrap
  interval, as for 1X2) and of `p_btts` against the outcome (Football-Data
  carries no BTTS prices).
* **Betting:** highest-EV selection per match across the markets in
  `--markets` (home/draw/away at the 1X2 prices, over/under at the totals
  prices) when `EV = P_model × odds − 1 > threshold`, fractional Kelly stakes
  on the start-of-day bankroll with per-bet and per-day caps. Reported: ROI,
  drawdown, bets and ROI by market and by selection, and flat-stake ROI with a
  bootstrap interval and `p_flat_roi_le_zero`.
* **Odds used:** the reference price is Pinnacle closing, then market-average
  closing, then Bet365 closing, then pre-match prices, for both markets; it is
  what forecasts are scored against. Over/under closing prices exist from
  2019/20. Market probabilities are de-margined with the power method
  (`implied_probabilities(..., "power")`; Shin's method is also available):
  proportional scaling leaves longshots with too much probability, which made
  them look like value when they were not.
* **Fair value and price** (`--fair`, `--price`): the EV can be judged with the
  model, with the de-margined closing line (the sharp price as fair value), or
  a blend, and the bet struck at the closing reference, the earliest listed
  price, or the best price across bookmakers (`max`). Every bet records its
  closing-line value, `p_closing × odds − 1`: the return the closing line
  expects for it. Average CLV is the yardstick that separates an edge from a
  lucky run; `--fair market --price max` is the pure price-shopping benchmark
  that needs no forecast at all.
* **Several seasons at once:** `--test-seasons 2021-2025` runs one continuous
  walk-forward over five seasons (about 1,900 scored matches) and reports the
  per-season breakdown too. One season of 380 matches cannot separate a
  0.002 Brier difference from noise; five can come close.

`--compare` reruns the season once per feature preset. Choose presets on an
earlier season (`--test-season` one year back) and score the final season only
once with the chosen setup. Otherwise the choice itself leaks test information.

## Results on real data

`make backtest-rolling` in October 2026: one walk-forward over 2021/22 to
2025/26 (1,900 Premier League matches, weekly refits, Premier League training
data only, ξ and xG weight tuned on 2019/20–2020/21, `lineup` features), scored
against Pinnacle or market-average closing prices.

| Market | Model Brier | Market Brier | Model minus market | 95% bootstrap interval |
| :--- | :--- | :--- | :--- | :--- |
| 1X2 (three-way Brier) | 0.579 | 0.567 | +0.012 | +0.007 to +0.017 |
| Over/under 2.5 (binary Brier) | 0.242 | 0.240 | +0.002 | −0.000 to +0.004 |
| Both teams to score (binary Brier, no market prices) | 0.246 | — | — | — |

| Season | 1X2 model | 1X2 market | Over 2.5 model | Over 2.5 market |
| :--- | :--- | :--- | :--- | :--- |
| 2021/22 | 0.565 | 0.554 | 0.242 | 0.247 |
| 2022/23 | 0.595 | 0.571 | 0.241 | 0.238 |
| 2023/24 | 0.544 | 0.526 | 0.232 | 0.227 |
| 2024/25 | 0.579 | 0.575 | 0.245 | 0.242 |
| 2025/26 | 0.612 | 0.609 | 0.249 | 0.247 |

What this says:

* **On match results the model is clearly behind the closing line.** One
  season (2025/26, +0.003) looked like parity; five seasons put the gap at
  +0.012 with an interval that excludes zero. The single-season numbers
  reported earlier were noise around that.
* **On goal totals it is level with the market.** The interval spans zero and
  the model beat the totals market in 2021/22. A goal-rate model is naturally
  stronger here than on the result, but "level" is not an edge.
* **No betting edge in either market.** Flat-stake ROI over 1,638 bets is
  −7.1% (interval −15.6% to +1.9%); −7.7% on 1X2 and −4.9% on totals. Treat the
  simulation as a test of edge rather than a strategy.
* **Features barely matter.** Over five seasons the presets sit within 0.002
  Brier of each other (`form` 0.577, `none` 0.578, `full` 0.578, `lineup`
  0.579). Line-up form helped on the three seasons where players have a year
  of history and hurt in 2022/23, its first season of data. Cup fixtures in
  the fatigue index and possession change nothing.
* **Hyperparameters barely matter** either: 64 settings within 0.005 Brier of
  each other on one season, and their ranking flips between seasons, which is
  why tuning uses two validation seasons.
* **More of the same data would not help.** At the tuned decay the fit
  effectively uses about 530 matches, 94% from the last two seasons.
  Training on both divisions made promoted clubs worse (0.624 against 0.614
  on 2025/26); recalibration, stacking with opening odds and betting filters
  did not move the result.

What the market knows and the model cannot see is squad change before kickoff:
transfers, injuries and the expected line-up. ESPN supplies the schedules and
possession FBref used to, and Understat line-ups stand in for SOFIFA ratings;
both are collected by `make data` and `make data-lineups`.

## Caveats

* The scrapers follow each site's markup as of writing. They are tested
  offline against fixtures that reproduce those formats
  (`tests/fake_sources.py`), not against the live sites. Run `make check` first:
  it shows exactly which source parses and which needs a selector update.
* SOFIFA and the anti-bot fallback need a Chromium that matches the installed
  Playwright (`make install-browser`), or pass `--chromium-path`.
* As of October 2026 FBref and SOFIFA answer every automated browser, headless
  or `--headful`, with a Cloudflare Turnstile challenge, so `make data` leaves
  them out. ESPN's public endpoints supply what FBref did (possession, shots,
  cup and European fixtures), and the lineup features fall back to Understat
  player form instead of SOFIFA ratings. `make check` shows when either site
  becomes reachable again.
* Lineups cost one request per match and SOFIFA one page load per 60 players
  per snapshot, so expect the first `make data-full` to take hours. Re-runs
  resume where they stopped.

## Tests

```bash
make test
```

The suite covers entity resolution, all parsers (including the Understat
JSON-endpoint fallback and Football-Data fixtures), anti-bot detection and
browser fallback, the full ETL → SQL → feature path on fake sources, schema
migration, SOFIFA resume, data-quality checks, the leakage audit, parameter
recovery, xG weighting and the promoted-club benefit on simulated leagues,
gradient checks, scoring rules, calibration, bootstrap intervals, Kelly
settlement, tuning and the CLI commands.
