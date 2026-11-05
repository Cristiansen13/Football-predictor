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
make data-leagues      # results and every bookmaker's prices for sixteen other leagues, 2019/20 on (price studies only)
make status            # coverage per season, data issues, leakage audit
make backtest          # tune ξ and xG weight on the two previous seasons, test on the last completed one
make backtest-rolling  # one walk-forward over 2021/22-2025/26: five seasons of scored matches, per-season table
make backtest-market   # price-shopping benchmark: the sharp closing line as fair value, bet the best price
make shop              # the same benchmark without a model, over the bookmakers the data names (BOOKS=max|named|soft|exchange, DIVISIONS=E0|all)
make predict           # this week's fixtures: probabilities, EV against current prices
```

Each target is a thin wrapper around the CLI:

| Command | Purpose |
| :--- | :--- |
| `football-predictor ingest [--seasons 2014-2025] [--sources …\|all] [--divisions E0 E1 …] [--lineups]` | ETL into `data/football.db`; exits non-zero if a source failed. `--divisions` takes any Football-Data league code (`D1`, `SP1`, `I1`, …; the model and the other sources cover `E0`/`E1` only) |
| `football-predictor check-sources [--season 2024]` | fetch and parse one live page per source |
| `football-predictor status [--audit N]` | data coverage report, quality issues, point-in-time audit of N matches |
| `football-predictor ratings [--as-of DATE]` | team attack/defence ratings |
| `football-predictor predict HOME AWAY` / `--upcoming DAYS` | model and market-anchored forecasts: score grids, markets, best-EV bet and stake |
| `football-predictor scan [--upcoming DAYS] [--fair anchored\|model] [--ev-threshold X] [--commission C]` | value board: every bookmaker's price on the coming fixtures (1X2, over/under, Asian handicap) against the fair value, with EV and Kelly stake; the exchange's prices net of commission |
| `football-predictor backtest [--test-season Y \| --test-seasons A-B] [--tune] [--compare] [--markets 1x2 totals] [--fair model\|market\|blend] [--price closing\|opening\|max] [--books named]` | walk-forward evaluation and betting simulation, 1X2 and over/under 2.5; `--books` takes the best price over chosen bookmakers instead of the market maximum |
| `football-predictor shop [--seasons A-B] [--divisions E0 …\|all] [--books max\|named\|soft\|exchange\|list] [--fair-book pinnacle] [--market-time closing\|opening] [--ev-threshold X] [--max-odds X] [--each-book]` | price shopping without a model: the sharp line as fair value, the bet wherever the chosen bookmakers beat it, at closing or at Football-Data's earlier prices; returns, bootstrap intervals and closing-line value by division, season, market and bookmaker |

Reports go to `data/reports/` (`metrics_<season>.json`, predictions, bets,
calibration table, PnL curve and reliability diagram). Use `--data-dir` to move them.

## 1. Data ingestion and entity resolution

| Source | Module | Method | Loaded into |
| :--- | :--- | :--- | :--- |
| Football-Data.co.uk | `etl/football_data.py` | direct CSV, Premier League (`E0`) and Championship (`E1`) by default, any of its other leagues on request (`--divisions`; `make data-leagues` loads sixteen), plus the weekly `fixtures.csv` | `matches`, `odds` (1X2 opening and closing prices of every bookmaker a file names: Bet365, bwin, Interwetten, William Hill, BetVictor, Ladbrokes, Betfair and its exchange, 1xBet, Betfred, BetMGM, Coral, Paddy Power, Sky Bet, Pinnacle, plus the market maximum and average), `totals_odds` (over/under 2.5), `handicap_odds` (Asian handicap line and prices), shot counts |
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
  spelling never creates a duplicate. Clubs outside England are known to
  Football-Data alone, so their spelling is registered as it is rather than
  matched: fuzzy matching would hand *Rangers* to Queens Park Rangers and
  *Dundee* to Dundee United, and refuse *Gent* next to *Genk*. Each club gets a deterministic UUIDv5,
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

**Market-anchored grid** (`models/anchored.py`, `DixonColesModel.predict_anchored`).
The closing line is the best pre-match forecast of the result, but it says
nothing directly about the score. The grid is the bridge: `rates_from_probabilities`
finds the two scoring rates whose Dixon-Coles grid reproduces the market's
home-win, away-win and over-2.5 probabilities (least squares when all three
are priced, exact when only 1X2 is), and the full correct-score distribution
follows. Fixtures without prices fall back to the model's own rates; each row
says which (`source`). `predict` prints both forecasts and both score grids,
and the backtest scores both (`anchored_*` metrics, `--anchor-price` picks
closing or opening prices).

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
  prices, both sides of the Asian handicap at its prices) when
  `EV = P × odds − 1 > threshold`, fractional Kelly stakes on the start-of-day
  bankroll with per-bet and per-day caps. Handicaps settle properly: a push
  returns the stake, a quarter line puts half the stake on each neighbouring
  half-goal line, and the probability used for EV counts a push as `1/odds`
  of a win (`handicap_effective_probabilities`). Reported: ROI, drawdown,
  bets and ROI by market and by selection, and flat-stake ROI with a
  bootstrap interval and `p_flat_roi_le_zero`.
* **Odds used:** the reference price is Pinnacle closing, then market-average
  closing, then Bet365 closing, then pre-match prices, for both markets; it is
  what forecasts are scored against. Over/under closing prices exist from
  2019/20. Market probabilities are de-margined with the power method
  (`implied_probabilities(..., "power")`; Shin's method is also available):
  proportional scaling leaves longshots with too much probability, which made
  them look like value when they were not.
* **Fair value and price** (`--fair`, `--price`): the EV can be judged with the
  model, with the de-margined closing line (the sharp price as fair value),
  with the market-anchored grid (`anchored`: the sharp line's own scoring
  rates, so 1X2, totals and handicaps are all priced from one consistent
  distribution), or a blend, and the bet struck at the closing reference, the
  earliest listed price, or the best price across bookmakers (`max`). Every bet records its
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

Predicting the winner and the score, same run, with the market-anchored
forecast alongside (rates that reproduce the closing prices through the grid):

| Forecast | Winner accuracy | Winner Brier | Exact score right | Score log loss |
| :--- | :--- | :--- | :--- | :--- |
| Model | 53.6% | 0.579 | 11.3% | 2.969 |
| Market-anchored grid | 55.4% | 0.567 | 12.0% | 2.933 |
| Closing market (1X2 only) | 55.4% | 0.567 | — | — |
| Historical frequencies 2014–2020 | 44.2% | 0.646 | 11.1% (always 1-1) | 3.092 |

The anchored grid is the best forecast these data allow: the market's winner
probabilities (reproducing them costs 0.0002 Brier) and a score distribution
that names the right score 12% of the time against the model's 11.3% and the
11.1% of always saying 1-1. Exact scores are mostly luck: a calibrated
forecast tops out near 12–13% on its most likely score, and the model's value
is in the whole distribution (the actual score is in its top five 47% of the
time). The model remains the forecast for fixtures without prices and the
independent second opinion next to the market's.

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

### Where an edge hides

If the forecast cannot beat the closing line, an edge can only come from
*when* and *where* a bet is placed. Seven seasons (2019/20–2025/26, 2,660
matches) with the model's walk-forward forecasts and every bookmaker's
opening and closing prices:

* **Betting earlier does not help.** The model trails even the opening line
  (1X2 Brier 0.581 against Pinnacle's 0.571 opening and 0.568 closing; over
  2.5: 0.243 against 0.240). Its disagreement with the opening 1X2 line says
  nothing about where the line closes (correlation 0.03); on totals it does a
  little (0.09), but the closing line still expects every model bet at
  opening prices to lose about 2% (negative closing-line value), and the
  realised returns agree: 1X2 +1% ± 9%, totals −4% ± 5%.
* **The model adds nothing to price shopping.** With the best market price
  as the bet, judging value by the model (+1.0%, CLV +0.3%) or a blend with
  the closing line (+1.7%, CLV +1.1%) is worse than using the closing line
  alone.
* **Price shopping against the sharp line looked like the one positive result.**
  Take Pinnacle's closing price, de-margined with the power method, as fair
  value, and bet whichever bookmaker offers a better price (`make
  backtest-market`, i.e. `--fair market --price max`, where the best price is
  Football-Data's market-maximum column):

  | EV threshold | Bets | Flat-stake ROI | 95% interval | Mean CLV | p(ROI ≤ 0) |
  | :--- | :--- | :--- | :--- | :--- | :--- |
  | 0 | 2,458 | +6.1% | +0.2% to +12.2% | +4.2% | 0.02 |
  | 2% | 1,800 | +6.9% | −0.9% to +14.6% | +5.4% | 0.04 |
  | 5% | 770 | +4.8% | −8.5% to +18.2% | +8.1% | 0.25 |

  By season at threshold 0: +27%, +5%, +12%, +1%, −4%, +3%, −1%, with
  closing-line value falling from +4.4% to +2.3%. The realised return tracks
  the CLV, which is what a genuine price discrepancy looks like and what the
  literature reports for this strategy. Three caveats: the "best price" is a
  market maximum over every bookmaker Football-Data's provider follows, most
  of which the files do not name (the bullet on bookmakers below takes this
  apart), bookmakers restrict accounts that do this, and the discrepancies
  are shrinking season by season. It is an edge in the market's own numbers,
  not in the model's.
* **Line-up news moves the line, but not enough to bet on.** The one feature
  that helped the forecast was the starting XI, so the natural question is
  whether its news is worth money in the hour between team sheets and
  kickoff. For 1,520 matches (2022/23–2025/26) the model predicted each
  match twice, with the XI that started and with the club's regular XI; the
  difference is the line-up surprise as the model reads it. It does point
  the way the market moves (correlation with the opening-to-closing move
  0.09 on 1X2, 0.22 on over/under, 0.18 in the top quintile of surprises;
  a side starting at under 85% of its regular XI is marked down about 1.8
  points by closing). But the market moves only a tenth to a sixth as far as
  the model's adjustment; adding the surprise to the opening line makes the
  forecast worse, and even scaled to a fifth it improves the opening line by
  0.0001 Brier; and bets in the surprise direction carry negative
  closing-line value (−2% to −4%) even at the opening price, which is days
  before anyone could act on a team sheet. The information is real and
  already priced. No timed line-up strategy was built.
* **The optimal bet: which market, which price, how much.** With Asian
  handicaps added (`handicap_odds`, Pinnacle margin 2.1%, half the lines are
  quarter lines) the engine scans 1X2, over/under 2.5 and both handicap
  sides, takes the best price across bookmakers and stakes a quarter Kelly
  capped at 5% of the bankroll per bet. Seven seasons, 2019/20–2025/26,
  EV threshold 2% (`make backtest-market`):

  | Fair value for the EV | Bets | Flat-stake ROI | 95% interval | Mean CLV | Kelly bankroll 1,000 → | Max drawdown |
  | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
  | Each market's sharp closing price (`--fair market`) | 1,909 | +5.7% | −1.1% to +12.9% | +5.5% | 1,962 | 35% |
  | Market-anchored grid (`--fair anchored`) | 2,294 | +2.1% | −4.3% to +8.8% | +3.8% | 1,086 | 49% |
  | Model (`--fair model`, EV 5%) | 2,552 | +1.7% | −6.0% to +9.0% | +0.3% | 122 | 96% |

  By market with the sharp price as fair: 1X2 +5.8% on 1,159 bets, over/under
  +9.2% on 389, handicap +1.7% on 361 (closing-line value +5.2%). The
  market-anchored grid, the best *forecast* of winner and score, is a worse
  judge of *value*: one score distribution cannot reproduce three markets'
  prices exactly, and the compromise shows up as false value, most of all on
  handicaps (−4.5% on 442 bets). The model as fair value finds no value at
  all, and Kelly staking on its overconfident probabilities loses 88% of the
  bankroll. So the answer to "which bet, when and how" is: judge each market
  by its own sharp closing price, bet whichever bookmaker beats it by at least
  2% at closing time, stake a quarter Kelly, and expect about 4–6% per bet
  before account limits, provided the better price is at a bookmaker one can
  hold an account with, which the next bullet shows is mostly not so. The
  caveats above still apply, and 2025/26 was negative in every configuration
  (−21% on 84 bets), in line with the shrinking discrepancies.
  `football-predictor scan` is the live version of this board for the coming
  fixtures, once Football-Data lists their prices.
* **Which bookmaker, really? The best price is mostly at bookmakers the data
  does not name.** The results above strike the bet at Football-Data's
  market-maximum column. The files also carry each listed bookmaker's own
  opening and closing prices (six in 2019/20–2023/24: Bet365, bwin,
  Interwetten, William Hill, BetVictor and Pinnacle; Betfair, its exchange
  and 1xBet in 2024/25; Betfred, BetMGM, Coral and Ladbrokes in 2025/26),
  now all loaded, and `football-predictor shop` evaluates the same strategy
  from the prices alone, with no model and over any set of bookmakers
  (`make shop`). It reproduces the benchmark exactly (same bets, same return)
  and then re-runs it on prices a bettor can actually take. Same seven
  seasons, same rule (Pinnacle closing as fair value, bet the best price
  when it beats it by 2%, flat stakes):

  | Prices shopped | Bets | Flat-stake ROI | 95% interval | Mean CLV |
  | :--- | :--- | :--- | :--- | :--- |
  | Market maximum (the benchmark above) | 1,909 | +5.7% | −1.6% to +12.8% | +5.5% |
  | Every bookmaker the file names | 471 | −2.2% | −17.6% to +13.7% | +4.6% |
  | The named bookmakers without Pinnacle and the exchange | 448 | −2.4% | −17.2% to +13.3% | +4.5% |
  | Betfair Exchange alone, 2024/25–2025/26, 5% commission, any positive EV | 160 | +1.0% | −23.2% to +27.9% | +2.4% |
  | Betfair Exchange alone, 2% commission | 487 | −1.0% | −14.1% to +10.9% | +1.8% |
  | Named soft books against the exchange's closing price as fair value, 2024/25–2025/26 | 83 | +4.0% | −27.5% to +36.4% | +3.3% |

  Where the maximum comes from, season by season, on 1X2 closing prices:

  | Season | Named bookmakers | Maximum above the best named price | Maximum from an unlisted bookmaker | Fixtures where the maximum beats Pinnacle by 2% | … where a named bookmaker does |
  | :--- | :--- | :--- | :--- | :--- | :--- |
  | 2019/20 | 6 | +2.9% | 84% | 68% | 25% |
  | 2020/21 | 6 | +3.1% | 89% | 72% | 14% |
  | 2021/22 | 6 | +3.6% | 88% | 82% | 23% |
  | 2022/23 | 6 | +3.3% | 87% | 67% | 10% |
  | 2023/24 | 6 | +3.5% | 86% | 66% | 10% |
  | 2024/25 | 7 | −0.2% | 29% | 42% | 33% |
  | 2025/26 | 9 | −1.9% | 8% | 17% | 30% |

  For five seasons the price the strategy took was, six times in seven, at a
  bookmaker the file does not name, about 3% above the best one it does. On
  the named prices the rule finds a quarter as many bets, the return is
  negative with an interval wide enough for anything, and the realised
  return no longer tracks the closing-line value, which it must for a price
  discrepancy to be worth money. Lowering the threshold to any positive EV
  finds 1,281 named bets at +3.7% (−4.6% to +11.8%, CLV +2.2%); raising it to
  5% finds 121 at +7.7% (−28.8% to +45.4%): noise around a small positive
  CLV. No single bookmaker gives a significant result (Interwetten +9.8% on
  151 bets, bwin +7.0% on 59, Bet365 −13.7% on 81, William Hill −8.4% on 93,
  BetVictor −16.1% on 106, each with an interval about ±35 points wide; the
  value BetVictor and William Hill offer is on longshots at mean odds of 5–6,
  which lose). On the named prices the totals and handicap markets, where
  only Bet365 and the exchange are named besides Pinnacle, give 31 and 29
  bets at −18% and −19%. The exchange is no better: at 2% commission it beats
  Pinnacle's de-margined price on 64% of fixtures, but by 1.8% on average,
  and returns −1%.

  From 2024/25 the named set includes the exchange and the maximum is no
  longer above the named prices, so the market-maximum strategy becomes
  reachable just as it stops working: the share of fixtures where the
  maximum beats Pinnacle by 2% fell from 66–82% to 42% and then 17%, and the
  named strategy returned −42% on 63 bets in 2025/26 (−67% to −13%). The
  honest reading of the price-shopping result is therefore: the
  discrepancies were real and large at the long tail of bookmakers the data
  aggregates, they are not detectable at the bookmakers a bettor can name,
  and they are closing. `football-predictor backtest --price max --books
  named` runs the model-based engine, Kelly staking included, on the same
  reachable prices. Two leagues cannot settle a 2% question, so the next
  section asks it of eighteen.

  A by-product worth knowing: Football-Data dropped Pinnacle during 2025/26
  (210 of 380 fixtures priced) and lists it nowhere in 2026/27, so the
  reference price and the market-anchored forecast fall back to the market
  average. That costs nothing. On the 590 fixtures of 2024/25–2025/26 priced
  by all of them, the de-margined closing prices of Pinnacle, the Betfair
  Exchange, the market average and Bet365 forecast within 0.001 Brier of
  each other (0.581, 0.581, 0.580, 0.580; the market maximum 0.580), at
  listed margins of 3.0%, 0.6%, 4.6% and 5.6%. The sharp price is not a
  better forecast than the soft consensus; it is only a cheaper one.
* **De-margining matters for this.** With proportional de-margining the same
  strategy looked like +5.5% CLV and returned 0%: proportional scaling gives
  longshots too much probability, so they looked like value. The power (or
  Shin) method prices favourites higher and the realised return then matches
  the CLV. The backtest uses the power method throughout.

### Eighteen leagues: the edge is on favourites at soft bookmakers

On the Premier League alone the named-bookmaker result was a quarter as many
bets as the benchmark with an interval thirty points wide; the same rule on
the Championship looked the same. A 2% edge needs tens of thousands of bets
to show, so `make data-leagues` loads sixteen more leagues from Football-Data
(the lower English and Scottish divisions, the top two in Germany, Spain,
Italy and France, the Netherlands, Belgium, Portugal, Turkey and Greece), with
the same bookmakers in every file. Seven seasons, 2019/20–2025/26, 44,986
matches, 1X2 only, the sharp line as fair value and the bet at the best
closing price among the soft bookmakers the files name (Bet365, bwin,
Interwetten, William Hill, BetVictor, and from 2024/25 1xBet, Betfair,
Betfred, BetMGM, Coral, Ladbrokes, Paddy Power and Sky Bet: everyone except
Pinnacle and the exchange), one bet per fixture, flat stakes. Seven rules were
fixed before any of them was run; the Premier League and Championship were
the leagues the rule was developed on, so the other sixteen are reported
separately as the out-of-sample set (`make shop DIVISIONS=all BOOKS=soft`,
`make shop-favourites`):

| Rule (soft books against Pinnacle closing, 1X2) | Bets | Flat-stake ROI | 95% interval | Mean CLV | p(ROI ≤ 0) | Sixteen other leagues only |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Any positive EV, all odds | 17,048 | +1.9% | −0.1% to +4.2% | +2.8% | 0.03 | 14,730 bets, +1.8% (−0.6% to +4.2%) |
| EV above 2% | 7,159 | +3.5% | −0.3% to +7.4% | +5.6% | 0.03 | 6,385 bets, +4.1% (+0.3% to +8.3%) |
| **Odds up to 3.0, any positive EV** | **9,368** | **+2.7%** | **+0.8% to +4.8%** | **+2.4%** | **0.004** | **8,090 bets, +2.9% (+0.7% to +5.1%)** |
| Odds up to 3.0, EV above 2% | 3,389 | +3.6% | +0.1% to +7.0% | +5.3% | 0.02 | 3,058 bets, +3.5% (−0.1% to +7.3%) |
| At Football-Data's earlier prices (days before kickoff) | 12,558 | +1.3% | −1.2% to +3.9% | +0.9% (EV +2.0%) | 0.16 | 11,179 bets, +1.5% |
| Market average as fair value instead of Pinnacle | 10,768 | +1.9% | −1.0% to +4.6% | +2.8% | 0.09 | 9,410 bets, +2.2% |
| Every named bookmaker, Pinnacle and the exchange included | 18,033 | +1.3% | −0.8% to +3.4% | +2.9% | 0.13 | 15,659 bets, +1.1% |

What holds up:

* **A small edge is real at soft bookmakers.** Every rule struck at closing
  returns more than zero (the all-odds rule +1.9% over seventeen thousand
  bets), and its closing-line value is between +2% and +5% in every league
  and between +2% and +4% in every season. The realised return only tracks
  it on favourites. By price band of the all-odds rule: odds up to 1.5
  return +3.7% (+1.5% to +6.0%, 2,302 bets) on +1.5% CLV, 1.5 to 2.5 about
  +3.4% on +2.7%, 2.5 to 4 about +0.3% on +2.7% CLV (7,103 bets), and above
  4 the CLV of +4% to +5% comes back as noise (2,807 bets, +2%, −7% to
  +11%). The de-margined sharp price still gives longshots more chance than
  they have, so a soft bookmaker's generous longshot price is not value; on
  favourites the sharp price is honest and the discrepancy is money. That is
  the favourite-longshot bias of the literature, seen from the other side,
  and the reason the favourites rule was on the list before the data came in.
* **The favourites rule survives every check tried.** Only fixtures Pinnacle
  itself closes: 8,958 bets, +2.9% (+0.8% to +4.9%). Four accounts only
  (Bet365, William Hill, bwin, Interwetten, the four named in every season):
  7,655 bets, +2.8% (+0.6% to +5.2%). Odds up to 2: 5,118 bets, +2.5% (+0.6%
  to +4.5%). EV above 1%: +2.8%; every qualifying selection rather than one
  per fixture: +2.8%. Home favourites +2.7% on 6,276 bets, away favourites
  +3.3% on 2,597. Seven rules were tried on the pooled data; the favourites
  rule's p-value of 0.004 is 0.025 after correcting for all seven, and it
  replicates on the sixteen leagues it was not developed on.
* **Season by season, and league by league, it is a thin edge:**

  | Season | Bets | Flat-stake ROI | 95% interval | Mean CLV |
  | :--- | :--- | :--- | :--- | :--- |
  | 2019/20 | 1,446 | 0.0% | −5.2% to +5.4% | +2.5% |
  | 2020/21 | 1,349 | +2.2% | −3.5% to +7.7% | +2.5% |
  | 2021/22 | 1,784 | +3.4% | −1.4% to +7.8% | +3.3% |
  | 2022/23 | 1,196 | +9.6% | +4.1% to +15.2% | +2.3% |
  | 2023/24 | 974 | +2.8% | −3.1% to +8.7% | +2.0% |
  | 2024/25 | 1,611 | +1.7% | −3.0% to +6.3% | +1.8% |
  | 2025/26 | 1,008 | −0.1% | −6.0% to +5.7% | +2.2% |

  Positive in thirteen leagues of eighteen (Belgium +9.6% on 633 bets,
  League One +9.0%, League Two +8.4%, Turkey +7.4%, Spain +6.6%; France
  −7.7% and the 2. Bundesliga −5.2% the worst), with the closing-line value
  steady at +1.6% to +4.7% everywhere, so a league's or a season's sign is
  mostly noise around a +2% to +3% expectation. Two seasons in seven came
  out flat and one carried nearly half of the profit. At 1% of the bankroll per bet,
  about 1,300 bets a season, that expectation is roughly +35% a season before
  account limits, and the limits are the real constraint: this is exactly
  the behaviour soft bookmakers restrict.
* **Where the price comes from.** William Hill supplies the most value
  (1,926 of the bets, +4.9% on +4.5% CLV), then Bet365 (+5.0% on 1,039),
  BetVictor (+3.2%), Interwetten (+2.2% on 2,274), bwin (+1.2% on 1,668) and
  1xBet (+0.7%); the bookmakers added in 2025/26 have too few bets to say.
  The Betfair Exchange, which does not restrict winners, gives +1.9% (−2.1%
  to +5.9%) on 2,369 favourites at 2% commission over its two seasons and
  loses at 5%; Pinnacle's closing price as the fair value is what makes the
  rule work, and judging the soft prices by the exchange's own closing price
  instead finds +1.4% on 2,000 bets.
* **Betting earlier does not help here either.** At the earlier prices the
  soft bookmakers beat Pinnacle's own earlier price by +2.0% on average, but
  by closing Pinnacle has moved towards them and the closing-line value of
  those bets is +0.9%. The sharp line moves and the soft line stays; the
  edge is in the soft line's staleness at closing, not in being early.

The practical reading: judge a 1X2 price by Pinnacle's de-margined closing
price, bet favourites (odds up to 3) wherever a soft bookmaker beats it, at
closing, flat stakes, across as many leagues as one holds accounts for, and
expect +2% to +3% per bet with seasons that can come out flat. The caveats
are the usual two and one new one: the prices are closing prices, reachable
only in the last minutes before kickoff; bookmakers restrict accounts that do
this; and Football-Data no longer lists Pinnacle, so a live version of the
rule needs the sharp price from somewhere else (`scan` falls back to the
market average, which forecasts as well but moves the thresholds).

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
* Football-Data's 2026/27 files carry no Pinnacle prices (and 2025/26 only
  for 210 fixtures), so the closing reference, the market-anchored forecast
  and the `scan` fair value use the market average there, which forecasts as
  well (see the bookmaker bullet above); `shop --fair-book betfair_exchange`
  judges the same strategy against the exchange's closing price instead.
  Which bookmakers a season file names changes every year or two; the loader
  takes whatever columns are present (`BOOKMAKERS` in `etl/football_data.py`).

## Tests

```bash
make test
```

The suite covers entity resolution (including exact registration of clubs
outside England), all parsers (including the Understat JSON-endpoint
fallback, Football-Data fixtures, every bookmaker code its files have used
and its stray zero prices), anti-bot detection and browser fallback, the full ETL →
SQL → feature path on fake sources, schema migration, SOFIFA resume,
data-quality checks, the leakage audit, parameter recovery, xG weighting and
the promoted-club benefit on simulated leagues, gradient checks, scoring
rules, calibration, bootstrap intervals, Kelly settlement, price shopping
(best prices over bookmaker sets, exchange commission, fair-value fallback,
closing-line value of bets struck earlier, several divisions at once,
settlement of 1X2, totals and handicap bets), tuning and the CLI commands.
