# Predictive Modeling of Premier League Match Outcomes: A Bivariate Poisson Framework

## 1. Data Ingestion & Entity Resolution (ETL Pipeline)

The foundation of a production-grade predictive model is a robust, automated ETL (Extract, Transform, Load) pipeline that aggregates historical statistics, advanced metrics, and market data. 

| Data Domain | Source | Extraction Methodology | Target Variables |
| :--- | :--- | :--- | :--- |
| **Match Events & Outcomes** | FBref | `BeautifulSoup` / `pandas.read_html` | Match results, possession, basic offensive/defensive metrics. |
| **Advanced Analytics** | Understat | Script tag JSON parsing via Regex | Expected Goals (xG), Expected Assists (xA), PPDA (pressing intensity). |
| **Squad Strength** | EA FC / SOFIFA | Dynamic scraping (`Playwright`) | Starting XI overall ratings (OVR) to quantify point-in-time team strength. |
| **Market Odds** | Football-Data.co.uk | Direct CSV ingestion | Historical closing odds for quantitative backtesting. |

**Entity Resolution (Master Data Management):**
A critical preprocessing step is reconciling disparate naming conventions across sources (e.g., "Man Utd", "Manchester United", "Manchester Utd"). The pipeline must implement a fuzzy matching algorithm (using Levenshtein distance) mapped to a standardized UUID schema to ensure data integrity during SQL `JOIN` operations.

---

## 2. Temporal Feature Engineering & State Representation

Traditional classifiers fail because they treat seasons as static datasets. A professional predictive model must represent the exact mathematical state of a team at the precise moment of kickoff. All engineered features must be strictly backward-looking to prevent data leakage.

*   **Exponentially Weighted Moving Averages (EWMA):** Simple rolling averages dilute recent form. Applying an EWMA to metrics like xG ensures that a match played four days ago carries a significantly higher mathematical weight than a match played a month ago.
*   **Point-in-Time Squad Fractional Value:** Teams are not static entities. By scraping the confirmed starting XI and aggregating their individual player ratings (OVR), the model calculates a `Lineup_Score`. This dynamically adjusts a team's offensive and defensive rating if key players are injured or suspended.
*   **Schedule Density & Fatigue Index:** Calculated as the inverse of `Days_Since_Last_Match`. A fatigue penalty function is applied dynamically if a team plays with less than 72 hours of rest, particularly after away fixtures.

---

## 3. The Probabilistic Engine (Dixon-Coles Model)

Rather than predicting categorical outcomes ($1$, $X$, $2$) directly using algorithms like Random Forest, the architecture utilizes a Bivariate Poisson regression framework optimized via Maximum Likelihood Estimation (MLE). This mimics the quantitative models utilized by professional syndicates.

### The Mathematical Framework
The model assumes that the number of goals scored by the Home team ($X$) and Away team ($Y$) are conditionally independent variables following Poisson distributions:

$$P(X=x) = \frac{e^{-\lambda} \lambda^x}{x!}, \quad P(Y=y) = \frac{e^{-\mu} \mu^y}{y!}$$

The scoring rates ($\lambda$ and $\mu$) are modeled log-linearly based on the engineered features:
*   $\lambda$ (Home Expected Goals) is a function of the Home Attack Strength, Away Defense Weakness, and the Home Advantage coefficient.
*   $\mu$ (Away Expected Goals) is a function of the Away Attack Strength and Home Defense Weakness.

### The Dixon-Coles Adjustments
A standard independent Poisson model underestimates the frequency of low-scoring draws ($0-0$, $1-1$). The architecture implements the Dixon-Coles $\rho$ (rho) parameter to correct this interdependence, modifying the joint probability function for low scores.

Furthermore, a time-decay parameter ($\xi$) is integrated directly into the MLE objective function. The optimizer dynamically scales down the influence of historical matches, ensuring the derived attack/defense coefficients reflect current reality.

---

## 4. Probability Matrix Generation

Once the optimal parameters are estimated, predicting a future fixture involves generating a complete probability matrix:

1.  **Rate Calculation:** Compute the expected goal rates ($\lambda$, $\mu$) based on the point-in-time feature vectors of both teams.
2.  **Matrix Formulation:** Calculate the joint probability for every possible correct score from $0-0$ up to $6-6$.
3.  **Outcome Aggregation:** Sum the probabilities within the matrix to derive the traditional market probabilities:
    *   $P(\text{Home Win}) = \sum_{x > y} P(X=x, Y=y)$
    *   $P(\text{Draw}) = \sum_{x = y} P(X=x, Y=y)$
    *   $P(\text{Away Win}) = \sum_{x < y} P(X=x, Y=y)$

---

## 5. Model Validation & Quantitative Backtesting

Standard academic metrics (Accuracy, F1-Score) are inadequate for stochastic sports environments. The model's validity must be proven through financial and probabilistic evaluation.

*   **Brier Score Evaluation:** The primary loss function utilized. It measures the mean squared difference between the predicted probability and the actual outcome, heavily penalizing overconfident, incorrect predictions.
*   **Expected Value (EV) Identification:** The aggregated probabilities are cross-referenced against historical closing odds. Positive expected value is identified where:
    $$EV = (P_{\text{model}} \times \text{Decimal\_Odds}) - 1 > 0$$
*   **Algorithmic Backtesting:** A simulation engine iterates over the temporal test set (e.g., the final season of the dataset). It executes virtual wagers solely on identified EV thresholds, scaling stakes dynamically using a fractional Kelly Criterion. A plotted Cumulative Profit & Loss (PnL) curve serves as the ultimate proof of the model's predictive edge over the bookmaker's margin.