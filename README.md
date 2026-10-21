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
make data              # results + odds (PL and Championship), xG/PPDA
make data-fbref        # + possession and cup fixtures (FBref; see the caveat on anti-bot challenges)
make data-full         # + starting XIs and SOFIFA ratings (slow on the first run, resumable; same caveat)
make status            # coverage per season, data issues, leakage audit
make backtest          # tune ξ and xG weight on the previous season, test on the last one
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
| `football-predictor backtest [--test-season Y] [--tune] [--compare]` | walk-forward evaluation and betting simulation |

Reports go to `data/reports/` (`metrics_<season>.json`, predictions, bets,
calibration table, PnL curve and reliability diagram). Use `--data-dir` to move them.

## 1. Data ingestion and entity resolution

| Source | Module | Method | Loaded into |
| :--- | :--- | :--- | :--- |
| Football-Data.co.uk | `etl/football_data.py` | direct CSV, Premier League (`E0`) and Championship (`E1`), plus the weekly `fixtures.csv` | `matches`, `odds` (opening/closing per bookmaker), shot counts |
| Understat | `etl/understat.py` | regex over `JSON.parse('\x..')` script tags; falls back to its JSON endpoint | `advanced_stats` (xG, npxG, PPDA, deep completions; xA from match rosters), `lineups` |
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
  squad's best XI.

## 3. Dixon-Coles model

```
log λ = c[div] + γ[div] + attack[home] + defence[away] + θ_att·A(home) + θ_def·D(away)
log μ = c[div]          + attack[away] + defence[home] + θ_att·A(away) + θ_def·D(home)
```

* **Dixon-Coles core.** The ρ correction applies to 0-0, 1-0, 0-1 and 1-1, and
  each match is weighted by `exp(−ξ·days)`. The fit is L-BFGS-B with an analytic
  gradient (checked against finite differences) and warm starts between refits.
* **Premier League and Championship on one scale.** Clubs carry their ratings
  through promotion and relegation, while each division keeps its own scoring
  level and home advantage. The ridge penalty shrinks each club towards its own
  division's mean. Within-division matches can't reveal how strong a division
  is as a whole, so a single shared target would overrate Championship clubs.
  On simulated leagues this halves the rating error for newly promoted clubs
  and lowers their early-season Brier score.
* **xG in the likelihood** (`--xg-weight`, CLI default 0.3). Each match
  contributes `(1−a)·log L(goals) + a·log L(xG)`. Ratings settle faster because
  xG is far less noisy than goals. Matches without xG (the Championship) use
  goals only.
* **Optional covariates.** Standardised features of the scoring side (A) and
  the conceding side (D). Presets: `none`, `form` (EWMA xG) and `full` (adds
  fatigue and lineup fraction). `--attack-features` / `--defence-features`
  override them.
* **Tuning.** `backtest --tune` grid-searches ξ × xG weight by walk-forward
  Brier score on the season before the test season.

## 4. Probability matrix

`score_matrices` builds the 7×7 joint grid. Mass beyond six goals is
renormalised away, so the 1X2 probabilities sum to one. `market_probabilities`
aggregates the grid into home/draw/away, over/under 2.5 and both-teams-to-score.

## 5. Validation and backtesting

`run_backtest` refits every `--refit-days` using only matches before each block.
It trains on every loaded division and scores only the Premier League. Metrics:

* **Forecast quality:** Brier (primary), RPS, log loss, exact-score log loss,
  accuracy, expected calibration error, and Brier skill against the historical
  base rates.
* **Against the market:** the same scores for the de-margined closing prices,
  Brier skill against the market, and a paired-bootstrap 95% interval for
  "model Brier − market Brier". An edge is only credible if the interval stays
  below zero.
* **Betting:** highest-EV outcome when `EV = P_model × odds − 1 > threshold`,
  fractional Kelly stakes on the start-of-day bankroll with per-bet and per-day
  caps. Reported: ROI, drawdown, ROI by selection (home/draw/away), and
  flat-stake ROI with a bootstrap interval and `p_flat_roi_le_zero`.
* **Odds used:** Pinnacle closing, then market-average closing, then Bet365
  closing, then pre-match prices. The best-of-market "max" price is never used.

`--compare` reruns the season once per feature preset. Choose presets on an
earlier season (`--test-season` one year back) and score the final season only
once with the chosen setup. Otherwise the choice itself leaks test information.

## Caveats

* The scrapers follow each site's markup as of writing. They are tested
  offline against fixtures that reproduce those formats
  (`tests/fake_sources.py`), not against the live sites. Run `make check` first:
  it shows exactly which source parses and which needs a selector update.
* SOFIFA and the anti-bot fallback need a Chromium that matches the installed
  Playwright (`make install-browser`), or pass `--chromium-path`.
* As of October 2026 FBref and SOFIFA answer every automated browser, headless
  or `--headful`, with a Cloudflare Turnstile challenge, so `make data` leaves
  them out. Football-Data and Understat cover everything the `none` and `form`
  presets need; without FBref the fatigue index ignores cup fixtures, and
  without SOFIFA the `full` preset has no lineup strength. `make check` shows
  when either site becomes reachable again.
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
