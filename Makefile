# Common workflows. Override variables on the command line, e.g.
#   make backtest TEST_SEASON=2023
PY ?= python3
CLI ?= football-predictor
TEST_SEASON ?=
SEASON_ARG = $(if $(TEST_SEASON),--test-season $(TEST_SEASON),)

.PHONY: install install-browser test demo check data data-full status backtest predict

install:            ## package + test dependencies
	$(PY) -m pip install -e ".[dev,scraping]"

install-browser:    ## Chromium for SOFIFA and for sites that block plain HTTP clients
	$(PY) -m playwright install chromium

test:
	$(PY) -m pytest

demo:               ## full backtest on a simulated league, no network
	$(CLI) demo --features form --tune --compare

check:              ## is every source reachable and parseable right now?
	$(CLI) check-sources

data:               ## results, odds (PL + Championship), xG/PPDA, possession, cup fixtures (ESPN)
	$(CLI) ingest --sources football_data understat espn

data-fbref:         ## possession and cup fixtures; FBref currently serves a Cloudflare challenge to browsers
	$(CLI) ingest --sources fbref

data-lineups:       ## starting XIs with each player's xG/xA (one request per match; resumable)
	$(CLI) ingest --sources understat --lineups --seasons 2022-$(shell date +%Y)

data-full: data data-lineups  ## + SOFIFA ratings (also behind Cloudflare at the moment)
	$(CLI) ingest --sources sofifa

status:             ## coverage per season, data issues, leakage audit
	$(CLI) status --audit 20

backtest:           ## tune on the two previous seasons, test on the last completed one
	$(CLI) backtest --tune --features lineup --compare $(SEASON_ARG)

TEST_SEASONS ?= 2021-2025
backtest-rolling:   ## one walk-forward across several seasons, reported overall and per season
	$(CLI) backtest --tune --features lineup --test-seasons $(TEST_SEASONS)

backtest-market:    ## price-shopping benchmark: Pinnacle closing as fair value, bet the best price across bookmakers
	$(CLI) backtest --features lineup --xi 0.003 --xg-weight 0.6 --fair market --price max --ev-threshold 0.02 --test-seasons 2019-2025

BOOKS ?= named
DIVISIONS ?= E0
shop:               ## the same benchmark without a model, over the bookmakers Football-Data names (BOOKS=max|named|soft|exchange|list, DIVISIONS=E0|all|codes)
	$(CLI) shop --seasons 2019-2025 --divisions $(DIVISIONS) --books $(BOOKS) --ev-threshold 0.02 --each-book

shop-favourites:    ## the one rule that held up across eighteen leagues: soft books against Pinnacle closing, odds up to 3, any positive EV
	$(CLI) shop --seasons 2019-2025 --divisions all --books soft --markets 1x2 --max-odds 3 --ev-threshold 0

LEAGUES ?= E2 E3 SC0 D1 D2 SP1 SP2 I1 I2 F1 F2 N1 B1 P1 T1 G1
data-leagues:       ## results and every bookmaker's prices for the other leagues Football-Data covers (no model data)
	$(CLI) ingest --sources football_data --divisions $(LEAGUES) --seasons 2019-$(shell date +%Y) --no-upcoming

log-prices:         ## log the coming fixtures' prices before kickoff (The Odds API key in ODDS_API_KEY or data/odds_api.key; one credit per league) and record the rule's bets
	$(CLI) log-prices

board:              ## the live value board from the logged prices: soft favourites against Pinnacle
	$(CLI) board

forward-test:       ## settle the ledger of paper bets against the results loaded (make data data-leagues) and the closing prices logged
	$(CLI) forward-test

predict:            ## this week's fixtures with probabilities and EV
	$(CLI) predict --upcoming 7 --features lineup
