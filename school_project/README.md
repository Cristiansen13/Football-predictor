# Predicting Premier League matches with a Poisson model

Course project for IAPD. Goal: forecast the result (home win / draw / away win)
and the score of Premier League matches, and measure honestly how good the
forecasts are, including against the bookmakers.

## What it does

1. **Data** (`footy/data.py`). One CSV per season from
   [Football-Data.co.uk](https://www.football-data.co.uk): every match with its
   final score and Bet365's odds (1X2 opening and closing, over/under 2.5 goals).
   `python main.py fetch` downloads the files into `data/`.
2. **Model** (`footy/model.py`). Goals follow a Poisson distribution whose mean
   depends on the league average for that venue, the attacking team's
   strength and the defending team's weakness:

   ```
   home goals ~ Poisson(home_attack × away_defence × league_home_average)
   away goals ~ Poisson(away_attack × home_defence × league_away_average)
   ```

   Attack strength is goals scored per game divided by the league average,
   defence weakness is goals conceded per game divided by the league average
   (1.0 = average). Strengths are smoothed as if every team had played five
   extra average games, so a club with a handful of matches is not rated on
   luck. The two Poisson variables are assumed independent, which makes the
   probability of any exact score a simple product. A 7×7 grid (0–6 goals
   each) gives the probabilities of each score and, by summing cells, of a
   home win, a draw, an away win, over/under 2.5 goals and both teams scoring.
3. **Evaluation** (`footy/evaluate.py`). Walk-forward: before every match day
   of the test season the model is refitted on the two previous seasons plus
   the test season so far, so no forecast ever uses information from the
   match itself or later. Forecasts are scored by accuracy, Brier score and
   log loss, the same scores are computed for the bookmaker from its closing
   odds (margin removed), and a flat-stake "value bet" rule (bet when
   `probability × odds − 1 > 5%`) shows whether beating the odds is realistic.

## How to run

```bash
pip install -r requirements.txt
python main.py fetch --seasons 2019-2025
python main.py backtest --test-season 2024
python main.py predict "Arsenal" "Chelsea"
python main.py table
python -m pytest tests
```

## Results

Three test seasons, 380 matches each, model refitted before every match day:

| Season | Winner accuracy, model | Winner accuracy, bookmaker | Brier, model | Brier, bookmaker | Exact score right | Always 1-1 right | Value bets: flat-stake ROI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2023/24 | 57.1% | 59.7% | 0.560 | 0.526 | 10.0% | 10.0% | −24.7% (306 bets) |
| 2024/25 | 51.1% | 55.5% | 0.597 | 0.576 | 11.3% | 11.8% | −16.0% (289 bets) |
| 2025/26 | 47.9% | 49.7% | 0.620 | 0.608 | 12.1% | 12.4% | −14.2% (277 bets) |

What the numbers say:

* **The model is a real forecaster but a worse one than the market.** It
  names the winner right about half the time, well above always picking the
  home side (41% in 2024/25), and its Brier score is 0.02 to 0.03 behind the
  bookmaker's every season. Bookmakers price in team news, injuries and
  line-ups that a goals-only model never sees.
* **Exact scores are mostly luck.** The most likely score comes up 10 to 12
  percent of the time, which is exactly how often always saying 1-1 would be
  right. The model's value is in the probabilities of all scores, not in
  naming one.
* **"Value bets" against the bookmaker lose money**, 14 to 25 percent of
  turnover. When a weaker forecast disagrees with a stronger one, the
  disagreement is mostly the weaker forecast's error, so the bets it finds
  are bad ones. This is the most important lesson of the project.

Example forecast (fitted on the last two seasons): Arsenal v Chelsea, expected
goals 1.77 – 0.96, home 56.5%, draw 23.3%, away 20.2%, over 2.5 goals 51.3%,
most likely score 1-0 (11.6%).

## What the full version adds

The `advanced-model` branch of this repository takes the same idea much
further: more data sources (expected goals, line-ups, cup schedules), a
time-weighted Dixon-Coles model, a market-anchored forecast that reaches the
bookmakers' accuracy, proper de-margining, Asian handicaps, closing-line
value, and a seven-season betting engine. Its conclusion matches the simple
version's: the forecast alone does not beat the closing line; the only
profitable approach found is comparing bookmakers' prices with the sharpest
one, which is a property of the market, not of the model.
