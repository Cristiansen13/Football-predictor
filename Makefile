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

predict:            ## this week's fixtures with probabilities and EV
	$(CLI) predict --upcoming 7 --features lineup
