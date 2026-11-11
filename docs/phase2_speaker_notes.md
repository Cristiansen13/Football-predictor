# Phase 2 presentation: speaker notes

Notes for [phase2_presentation.pptx](phase2_presentation.pptx). The same text is in the notes pane of every slide, so it appears in PowerPoint's presenter view.

The slot is 10 minutes: at most 7 minutes of presentation, then discussion. Read at a calm pace, these notes take about 6:15.

| Slide | Title | Time | Running total |
| ---: | :--- | ---: | ---: |
| 1 | Predicting Premier League match results | 0:15 | 0:15 |
| 2 | Home sides win more; goals follow Poisson | 0:40 | 0:55 |
| 3 | Long-run strength predicts results; form adds little | 0:45 | 1:40 |
| 4 | Six models, one walk-forward protocol | 0:55 | 2:35 |
| 5 | No model reaches the bookmaker | 0:55 | 3:30 |
| 6 | Draws are a quarter of results and never predicted | 0:45 | 4:15 |
| 7 | When they disagree, the bookmaker is right | 0:45 | 5:00 |
| 8 | Clubs new to the league are the blind spot | 0:40 | 5:40 |
| 9 | What we learned | 0:35 | 6:15 |

## Slide 1. Predicting Premier League match results

*About 15 seconds.*

Hello. In the first phase we presented the idea and the data. Today I will show what the exploratory analysis found, how six models compare with each other and with the bookmaker, and where their errors come from.

## Slide 2. Home sides win more; goals follow Poisson

*About 40 seconds.*

Two findings from the exploration. On the left, results by season. The home side wins 44 percent of matches, draws 24 and loses 32. The one exception is 2020/21, played mostly without crowds, where home wins fell to 38 percent. On the right, the goals scored by the home side, in blue, against a Poisson distribution with the same average, in grey. Every bar is within two percentage points. That justifies a Poisson goal model: the expected goals of the two sides are enough to give a probability to every score.

## Slide 3. Long-run strength predicts results; form adds little

*About 45 seconds.*

What information predicts the result? On the left we group matches by the difference in Elo rating between the home and the away side. When the home side is much weaker it wins 13 percent of the time; when it is much stronger, 78 percent. Look at the orange line, the draws: it stays near one in four whatever the difference. On the right, each bar is one feature and its correlation with the goal difference. The rating and the goals over a whole season are at the top. Form over the last five matches is weaker, and rest days carry nothing.

## Slide 4. Six models, one walk-forward protocol

*About 55 seconds.*

We compare six models. Class frequencies give the same three probabilities to every match; that is the floor. Elo uses a single feature, the rating difference. The Poisson goal model works from goals and gives a probability for every score. Then three classifiers on all 21 features: logistic regression, random forest and gradient boosting. The protocol matters more than the models. We use the seasons in time order: six for training, two validation seasons to choose the settings, and the last three as the test, 1,140 matches. Before each test season a model is refitted on everything earlier, so no forecast ever sees the future. We score with accuracy, Brier score and log loss, and the reference is Bet365's odds at kickoff.

## Slide 5. No model reaches the bookmaker

*About 55 seconds.*

This is the main result; a lower Brier score is better. The best of our models is logistic regression, with 54 percent accuracy and a Brier score of 0.584. The bookmaker, in the last row, has 55 percent and 0.570. The gap is small, 0.014, but it is real: the last column gives a bootstrap interval for the gap, and it excludes zero for every model. Two more things. First, Elo, with one single feature, does as well as logistic regression with 21. Second, the more flexible models, random forest and gradient boosting, are slightly worse, not better: with a few thousand matches and a very noisy target, the simple model wins. All of them are far better than class frequencies, at 0.650.

## Slide 6. Draws are a quarter of results and never predicted

*About 45 seconds.*

Now the error analysis. First, draws. These are confusion matrices. Rows are what happened, columns are the result the forecaster rated most likely. Look at the middle column, predicted draw: it is empty, for our model and for the bookmaker. 279 matches ended in a draw and not one was predicted. The highest draw probability our model ever gives is 30 percent, so a draw is never the most likely result. This is why nobody gets much above 55 percent accuracy. It is not a calibration problem: when the model says 30 percent, it happens about 30 percent of the time.

## Slide 7. When they disagree, the bookmaker is right

*About 45 seconds.*

Second, we compared our forecasts with the bookmaker's. On the left, matches are grouped by how far apart the two forecasts are. In half of the matches they are within five points, and they score the same. But in the 182 matches where they differ by more than ten points, the bookmaker is clearly better: 0.610 against 0.672. The table on the right says the same thing in money. If we bet whenever a model thinks the odds are too generous, every model loses, from 7 to more than 18 percent of the stakes. So when the two disagree, the model is wrong.

## Slide 8. Clubs new to the league are the blind spot

*About 40 seconds.*

Third, errors by club. Each bar is a club: the points per game it really won, minus what the model expected. Right means under-rated, left over-rated. The eight largest errors are all promoted clubs. Sunderland and Leeds did much better than expected; Southampton, Leicester and Luton much worse. The reason is that a promoted club arrives with no recent history in the league, so its rating is a guess. And the worst single misses are upsets, like Liverpool losing at home to Crystal Palace: 6 percent for our model, 9 for the bookmaker.

## Slide 9. What we learned

*About 35 seconds.*

To conclude. A simple, well calibrated model built on public results gets most of the way: 54 percent accuracy and a Brier score of 0.584, against 0.650 for class frequencies. The bookmaker stays ahead at 0.570, because the odds include information that results do not, such as line-ups and injuries. More complex models did not help. And the error analysis tells us what to try next: squad and line-up data, a model aimed at draws, and better starting ratings for promoted clubs. Thank you.

## Questions to expect

**Why is the simplest model the best? Why did boosting not win?**

A football result is a very noisy signal, we have three to four thousand training matches, and team strength is close to one number. A flexible model then mostly fits noise. The validation seasons agreed: for every classifier they chose the most regularised setting.

**How do you know the gap to the bookmaker is not chance?**

A paired bootstrap. We resample the same 1,140 matches 2,000 times and compute the difference in Brier score each time. For logistic regression the 95 percent interval is 0.007 to 0.021, so it excludes zero.

**Did you tune anything on the test seasons?**

No. Settings were chosen on two earlier validation seasons, 2021/22 and 2022/23, and then frozen. The test seasons were only used once.

**Why the Brier score and not just accuracy?**

Accuracy looks only at the most likely result, so it ignores how confident the forecast is and it never rewards a draw probability. The Brier score and the log loss judge the three probabilities.

**Why is every model worse in 2025/26 than in 2023/24?**

Seasons differ in how predictable they are. The bookmaker's score rises in the same way, from 0.526 to 0.608, so it is the season, not the models.

**Can a model ever predict a draw?**

Not as the most likely result: a home or an away win is always more probable. What can improve is the draw probability itself, for example with a Dixon-Coles correction to the Poisson model.

**What is the Poisson model good for, if it scores worst?**

It is the only one that gives a probability for every score, so it also answers questions such as over or under 2.5 goals. The classifiers only give the three results.

**How would you fix the promoted clubs?**

Give them a better starting rating, from their results in the division below or from the value of their squad, instead of the average of the relegated clubs.

**Would the odds as a feature beat the bookmaker?**

No. The model would learn to copy the odds, and the betting check shows what happens when we disagree with them: every model loses money.
