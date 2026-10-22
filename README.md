# Premier League match prediction

University project (Poli, IAPD): predicting the winner and the score of
Premier League matches with a Poisson goal model, and checking the forecasts
against the bookmakers.

The repository has two parts:

| Where | What |
| :--- | :--- |
| [`school_project/`](school_project/) on this branch (`master`) | The course deliverable: a short, readable pipeline. Match results and Bet365 odds from Football-Data.co.uk, a Poisson model built from team scoring averages, a 7×7 correct-score grid, a walk-forward evaluation against the bookmaker, and a simple value-betting check. |
| branch [`advanced-model`](../../tree/advanced-model) | The full research pipeline that grew out of the course work: four data sources (Football-Data, Understat, ESPN, SOFIFA), a time-weighted Dixon-Coles model with xG and line-up features, market-anchored forecasts, Asian handicap pricing, closing-line value, a multi-season betting engine and a live value board. Its README reports the results over seven seasons. |

[`project_architecture.md`](project_architecture.md) is the original design
document both parts follow.

## Quick start (school project)

```bash
cd school_project
pip install -r requirements.txt
python main.py fetch --seasons 2019-2025        # download the season files (one CSV each)
python main.py backtest --test-season 2024      # evaluate one season, refitting before every match day
python main.py predict "Arsenal" "Chelsea"       # probabilities and score grid for a fixture
python -m pytest tests                           # offline tests
```

See [`school_project/README.md`](school_project/README.md) for the method and the results.
