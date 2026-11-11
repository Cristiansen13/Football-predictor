# Phase 1 presentation: speaker notes

Notes for [phase1_presentation.pptx](phase1_presentation.pptx). The same text is in the notes pane of every slide, so it appears in PowerPoint's presenter view.

The slot is 10 minutes: 5 to 7 minutes of presentation, then discussion. Read at a calm pace, these notes take about 6:20.

| Slide | Title | Time | Running total |
| ---: | :--- | ---: | ---: |
| 1 | Predicting Premier League match results | 0:20 | 0:20 |
| 2 | Can public data forecast a match as well as the odds? | 1:05 | 1:25 |
| 3 | One CSV per season, fetched by a small crawler | 0:55 | 2:20 |
| 4 | Cleaning turns twelve raw files into two tables | 0:55 | 3:15 |
| 5 | The dataset at a glance | 0:35 | 3:50 |
| 6 | The home side wins 44% of matches | 0:45 | 4:35 |
| 7 | Goals follow a Poisson distribution closely | 0:45 | 5:20 |
| 8 | What we will do in the final phase | 1:00 | 6:20 |

## Slide 1. Predicting Premier League match results

*About 20 seconds.*

Hello. Our project is about forecasting the results of Premier League football matches. Today I will cover four things: the idea, how we collected the data, what the dataset looks like, and the analyses we plan for the final phase.

## Slide 2. Can public data forecast a match as well as the odds?

*About 65 seconds.*

The task is simple to state. Before a match starts, we want three probabilities: that the home side wins, that the match is drawn, and that the away side wins. So it is a classification problem with three classes, but we care about the probabilities, not only about the most likely class. Everything we use is public: twelve seasons of results, match statistics and bookmaker odds. The odds give us something most projects do not have: a very strong benchmark. A bookmaker's odds are a forecast too, so our question is not only whether a model beats chance, but how close it gets to the market. On the right is the kind of output we want, from a first prototype. For Arsenal against Chelsea it gives 56.5 percent for a home win, 23 for a draw and 20 for an away win.

## Slide 3. One CSV per season, fetched by a small crawler

*About 55 seconds.*

The data come from Football-Data.co.uk, a site that publishes one CSV file per league and per season. Our crawler is a small Python script. It builds the address of each season file from the season code, for example 2324 for the season 2023/24. It downloads the file once, checks that what came back really is a match file, and saves it untouched in the raw data folder. It is a polite crawler: it identifies itself, waits one second between requests, and skips files it already has, so we can run it again safely. The result is twelve files, one per season, each with 380 rows, one per match. The number of columns grows from 62 to 132 over the years, because the site lists more bookmakers.

## Slide 4. Cleaning turns twelve raw files into two tables

*About 55 seconds.*

Preprocessing turns the twelve raw files into two tables. The first is the match table, with one row per match. The site has used two date formats over the years, so we read both into one. We drop blank rows, keep 31 of up to 132 columns and give them readable names, and derive the result from the score. If a season does not have a column, we leave it empty; we never invent values. The second table is for modelling: 21 features per match that describe both teams before kickoff. Examples are an Elo rating, form over the last five matches, and days of rest. The important rule is that every feature uses only earlier matches, and we have a test that checks it.

## Slide 5. The dataset at a glance

*About 35 seconds.*

Here is the dataset in numbers. 4,560 matches from twelve complete seasons, 2014/15 to 2025/26, and 35 different clubs. There are 2.8 goals per match on average, and the home side wins 44 percent of matches. The table shows the six groups of columns. The data are complete, with one exception: the odds at kickoff and the over/under odds are only published from 2019/20, so they are missing for the first 1,900 matches. This is why our evaluation will use the most recent seasons.

## Slide 6. The home side wins 44% of matches

*About 45 seconds.*

The first pattern in the data is home advantage. Each bar is a season, split into home wins in blue, draws in orange and away wins in green. Over twelve seasons the home side wins 44 percent of matches, draws 24 and loses 32. There is one clear exception: 2020/21, the season played mostly in empty stadiums. Home wins fell to 38 percent and away wins rose to 40. So the crowd seems to matter. Notice also the draws: between 19 and 28 percent every season, about a quarter of all matches. They will be the hardest result to predict.

## Slide 7. Goals follow a Poisson distribution closely

*About 45 seconds.*

The second pattern motivates our first model. The blue bars show how often the home side scores zero, one, two goals and so on. The grey bars show a Poisson distribution with the same average, 1.54 goals. The two are very close. This means that if we can estimate the expected number of goals of each side, we get the probability of every possible score, and from the scores the probability of each result. On the right are the most frequent scores. The most common is 1-1, in about 11 percent of matches, so naming the exact score is mostly luck.

## Slide 8. What we will do in the final phase

*About 60 seconds.*

For the final phase we plan four steps. First, explore: home advantage over time, the goal distributions, and which of our features are related to the result. Second, model: we start with class frequencies as a floor, then an Elo rating, a Poisson goal model, and three machine learning classifiers: logistic regression, random forest and gradient boosting. Third, evaluate honestly: every model predicts the last three seasons, 1,140 matches, without ever seeing the future, and we score it by accuracy, Brier score and log loss against the bookmaker's odds at kickoff. Fourth, analyse the errors: confusion matrix, calibration, and errors by kind of match and by club. The repository already follows the required structure: extraction, data and docs, with analysis to come. Thank you. I am happy to take questions.

## Questions to expect

**Why this source, and not a bigger site?**

Football-Data.co.uk offers the files for download, in a stable format, and it is the only free source that has results, match statistics and bookmaker odds in one place. One source keeps the pipeline simple and reproducible.

**Is the crawling acceptable for the site?**

Yes. The files are published for download. We make twelve requests in total, one per season, with a pause between them, and we never download a file twice.

**What is an Elo rating?**

One number per club. After every match the winner's rating goes up and the loser's goes down, by more when the result was unexpected. It comes from chess.

**How do you avoid using information from the future?**

Twice. Every feature of a match is computed from earlier matches only, and a unit test checks it. And the evaluation is walk-forward: a model is trained on past seasons and predicts a later one.

**What is the Brier score?**

The mean squared error of the three probabilities. Zero is a perfect forecast and two is the worst. It rewards being right and being honest about uncertainty.

**Will you use the odds as a feature?**

No. The odds are the benchmark. If the model could see them it would mostly copy them, and we want to know what public results alone can do.

**Why only the Premier League, and why from 2014?**

To keep the scope of a course project. Twelve seasons give 4,560 matches, and the odds at kickoff that we need for the benchmark exist only from 2019/20.
