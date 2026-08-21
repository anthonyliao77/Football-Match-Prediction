# Football Match Prediction

A football match prediction project using historical match statistics, rolling team-form features, Understat expected goals, and Elo ratings to predict **Home Win, Draw, or Away Win**.

**Models:** Random Forest · XGBoost
**Leagues:** Premier League · La Liga · Serie A
**Prediction type:** Three-class match outcome probabilities

> This is a research-oriented football analytics project focused on machine learning, feature engineering, evaluation, and experimentation. It is not a production betting or live prediction system.

## Overview

The project follows a chronological football prediction pipeline:

1. Historical match data is loaded from local CSV files.
2. Understat schedule data is obtained through `soccerdata`.
3. The datasets are aligned by date and team names.
4. Historical rolling team statistics are calculated using previous matches.
5. Understat xG and xGA information is incorporated into rolling features.
6. Elo ratings are calculated sequentially so that each fixture receives the ratings available **before that match**.
7. Data is split chronologically by football season rather than randomly.
8. Random Forest and XGBoost classifiers are trained on historical seasons.
9. The models are evaluated on the latest held-out season.
10. Predictions include both the predicted outcome and probabilities for Home, Draw, and Away.

## Current Status

### Implemented

* [x] Historical football match data pipeline
* [x] Multiple league support
* [x] Understat xG integration
* [x] Rolling five-match team statistics
* [x] Rolling xG and xGA features
* [x] Sequential Elo rating system
* [x] Home advantage in Elo calculations
* [x] New-season Elo regression
* [x] Season-based chronological validation
* [x] Random Forest classifier
* [x] XGBoost classifier
* [x] Match outcome probabilities
* [x] Accuracy evaluation
* [x] Log loss evaluation
* [x] Multiclass Brier score evaluation

### Planned

* [ ] Poisson goal prediction model
* [ ] Scoreline probability predictions
* [ ] Walk-forward validation across multiple seasons
* [ ] Probability calibration
* [ ] Additional leagues and historical data
* [ ] Automated data refreshing
* [ ] Comparison between classification and goal-based prediction models

---

## Data Pipeline

The overall pipeline is:

```text
Historical Match Data
        │
        ├───────────────┐
        │               │
        ▼               ▼
Football-Data       Understat
        │               │
        └───────┬───────┘
                ▼
       Data Alignment
                │
                ▼
        Feature Engineering
                │
        ┌───────┴────────┐
        ▼                ▼
   Rolling Form        Elo
        │                │
        └───────┬────────┘
                ▼
       Season-Based Split
                │
        ┌───────┴────────┐
        ▼                ▼
     Training        Validation
        │                │
        └───────┬────────┘
                ▼
       Random Forest / XGBoost
                │
                ▼
       Outcome + Probabilities
```

---

## Features

The models use historical information that should be available before the predicted fixture.

### Recent Form

Rolling statistics are calculated from each team's previous five matches.

| Feature                 | Description                                          |
| ----------------------- | ---------------------------------------------------- |
| `HomePT5` / `AwayPT5`   | Points from the previous five matches                |
| `HomeGS5` / `AwayGS5`   | Goals scored in the previous five matches            |
| `HomeGC5` / `AwayGC5`   | Goals conceded in the previous five matches          |
| `HomeGD5` / `AwayGD5`   | Goal difference over the previous five matches       |
| `HomeSOT5` / `AwaySOT5` | Shots on target over the previous five matches       |
| `HomeS5` / `AwayS5`     | Total shots over the previous five matches           |
| `HomeSC5` / `AwaySC5`   | Shots conversion rate over the previous five matches |

The current fixture's result and match statistics are not included when calculating these rolling features.

For example, if predicting a match on Saturday, the rolling features represent information from matches that occurred **before Saturday's fixture**.

### Expected Goals

Historical expected-goals data is obtained from Understat through the `soccerdata` package.

The project uses:

* `home_xg`
* `away_xg`

These are incorporated into rolling five-match features:

| Feature                 | Description                                           |
| ----------------------- | ----------------------------------------------------- |
| `HomeXG5` / `AwayXG5`   | Expected goals over the previous five matches         |
| `HomeXGA5` / `AwayXGA5` | Expected goals against over the previous five matches |

Understat xG is currently used as a feature for the classification models. The project does **not** currently implement a separate Poisson goal model.

### Elo

The project calculates sequential Elo ratings for each team.

The model uses:

* `HomeEloBefore`
* `AwayEloBefore`

These represent the ratings immediately before the fixture.

The system also calculates post-match ratings internally, but the model does not use those post-match values as prediction features.

---

## Data Sources

### Football-Data

Historical match statistics are stored locally in:

```text
football_data/
├── LaLiga/
├── PremierLeague/
└── SerieA/
```

The data provides information such as:

* Match results
* Goals
* Shots
* Shots on target
* Other historical match statistics

The data is loaded, combined, normalized, and sorted chronologically before feature engineering.

No API key is required for these local files.

### Understat

Understat provides historical expected-goals information.

The project accesses Understat through the [`soccerdata`](https://github.com/amosbastian/soccerdata) Python package.

The Understat data is used to obtain:

* Match dates
* Home and away teams
* Home expected goals
* Away expected goals

Team names are mapped to the naming convention used by the local football-data files before merging.

No API key is currently used by this project for Understat.

---

## Data Preprocessing

### Date Handling

Match dates are converted to pandas datetime values and normalized before the datasets are sorted chronologically.

This ensures that feature engineering and Elo calculations process matches in temporal order.

### Team Names

Team names from different sources are normalized so that the same club can be matched across datasets.

Understat team names are mapped using the project's team-name mapping.

### Dataset Alignment

The local match data and Understat data are matched using the match date and home/away team names.

The current implementation uses a nearest-date merge with a one-day tolerance.

This makes the team-name mapping and date normalization important for successful matching.

### Missing Data

The current pipeline does not implement a general-purpose missing-value imputation system.

The project currently expects the supported source data to contain sufficient information for the selected seasons.

---

## Season-Based Validation

Football matches are time-dependent, so the project does **not** randomly shuffle matches before training.

Instead, matches are divided by football season.

A season is determined using a July boundary:

```text
July 2025 → 2025/26
June 2026 → 2025/26
July 2026 → 2026/27
```

The latest complete season is held out for validation.

For example:

```text
Training:
2020/21
2021/22
2022/23
2023/24
2024/25

Validation:
2025/26
```

This approach ensures that validation matches occur after the matches used to train the model.

It also provides a more realistic approximation of the real-world prediction problem:

```text
Past seasons
     ↓
Train model
     ↓
Upcoming season
     ↓
Make predictions
```

The project also verifies that training and validation seasons do not overlap.

---

## Elo Rating System

The project implements a sequential Elo rating system in `src/elo.py`.

### Configuration

The current configuration includes:

| Parameter          | Value | Description                                                      |
| ------------------ | ----: | ---------------------------------------------------------------- |
| `INITIAL_RATING`   |  1500 | Initial rating for new teams                                     |
| `K_FACTOR`         |    32 | Controls the size of rating updates                              |
| `HOME_ADVANTAGE`   |   100 | Elo points added to the home team for expected-score calculation |
| `A_FACTOR`         |  0.75 | New-season regression factor                                     |
| `NEW_SEASON_MONTH` |     7 | Month used to identify the beginning of a new season             |

These parameters are configurable and can be tested experimentally.

### Expected Score

The expected result is calculated using the standard Elo logistic formula:

[
E = \frac{1}{1 + 10^{(R_{opp}-R_{team})/400}}
]

For a home fixture, the home team's rating receives the configured home-advantage adjustment before calculating the expected score.

### Rating Update

After the match result is known, the rating is updated using:

[
R' = R + K(S-E)
]

where:

* (R) = current rating
* (R') = updated rating
* (K) = K-factor
* (S) = actual result
* (E) = expected result

The actual result is represented as:

```text
Win  → 1
Draw → 0.5
Loss → 0
```

### New-Season Regression

At the beginning of a new season, ratings are partially regressed toward the initial rating:

[
R_{new} =
A \times R_{old}
+
(1-A) \times R_{initial}
]

This allows previous-season strength to carry over while reducing the influence of outdated ratings.

### Preventing Elo Leakage

Elo is calculated sequentially.

For each fixture:

```text
1. Read current home and away Elo
2. Store HomeEloBefore / AwayEloBefore
3. Use those values for prediction features
4. Process the match result
5. Update both teams' Elo ratings
6. Move to the next match
```

Therefore, the current match result is not used to calculate the Elo feature for that same fixture.

---

## Machine Learning Models

### Random Forest

The Random Forest classifier predicts one of three classes:

```text
H = Home Win
D = Draw
A = Away Win
```

The current configuration is:

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=67,
    n_jobs=-1
)
```

The model generates:

* Class predictions using `predict()`
* Class probabilities using `predict_proba()`

### XGBoost

The project also uses `XGBClassifier` for three-class classification.

The current configuration is:

```python
XGBClassifier(
    n_estimators=200,
    random_state=67,
    learning_rate=0.01,
    n_jobs=-1
)
```

XGBoost requires the target classes to be represented numerically, so the result labels are encoded before training.

The original classes:

```text
A
D
H
```

are converted into numeric class labels using `LabelEncoder`.

The predictions are converted back to the original labels after prediction.

XGBoost also provides class probabilities through `predict_proba()`.

---

## Model Evaluation

The project evaluates both the predicted class and the quality of the predicted probabilities.

### Accuracy

Accuracy measures the percentage of validation matches where the predicted outcome is correct.

[
Accuracy =
\frac{\text{Correct Predictions}}
{\text{Total Predictions}}
]

Higher is better.

Accuracy is easy to understand, but it does not measure whether the predicted probabilities are well calibrated.

### Log Loss

Log loss evaluates the probability assigned to the actual outcome.

A confident incorrect prediction receives a much larger penalty than an uncertain incorrect prediction.

Lower values are better.

This makes log loss particularly useful for this project because the models produce probabilities rather than only class predictions.

### Multiclass Brier Score

The project calculates a multiclass Brier score by comparing the predicted probability vector with the one-hot encoded actual outcome.

For each match, the model produces:

```text
Home probability
Draw probability
Away probability
```

The score measures how close those probabilities are to the actual result.

Lower values are better.

### Confusion Matrix

A confusion matrix can also be used to examine which outcomes the model predicts correctly or incorrectly.

The main training output currently focuses on accuracy, log loss, and Brier score.

---

## Results

Model results are currently generated when the training pipeline is executed rather than stored as permanent benchmark files.

This means the repository does not currently claim a single fixed accuracy or probability score.

Results should be compared using:

* Accuracy
* Log loss
* Brier score

When benchmark experiments are formally recorded, results can be presented in a table such as:

| League         | Model         | Elo | Accuracy | Log Loss | Brier |
| -------------- | ------------- | --: | -------: | -------: | ----: |
| Premier League | Random Forest | Yes |        — |        — |     — |
| Premier League | XGBoost       | Yes |        — |        — |     — |
| La Liga        | Random Forest | Yes |        — |        — |     — |
| La Liga        | XGBoost       | Yes |        — |        — |     — |
| Serie A        | Random Forest | Yes |        — |        — |     — |
| Serie A        | XGBoost       | Yes |        — |        — |     — |

The results are intentionally not hardcoded into this README because they can change as features, Elo parameters, and model configurations are tested.

---

## Prediction Output

For each validation fixture, the models can produce:

* Date
* Home team
* Away team
* Actual result
* Predicted result
* Away probability
* Draw probability
* Home probability

Example:

```text
Match: Team A vs Team B

Prediction: Home Win

Away: 18%
Draw: 24%
Home: 58%
```

The probabilities represent the model's estimated probability of each outcome.

The current project predicts **match outcomes**, not final scorelines.

---

## Project Structure

```text
Football-Prediction-Model/
├── config.py
├── train.py
├── requirements.txt
├── football_data/
│   ├── LaLiga/
│   ├── PremierLeague/
│   └── SerieA/
├── src/
│   ├── data_loader.py
│   ├── elo.py
│   ├── features.py
│   ├── training.py
│   └── understat_loader.py
├── .gitignore
└── README.md
```

### Main Files

| File                      | Purpose                                                                             |
| ------------------------- | ----------------------------------------------------------------------------------- |
| `config.py`               | Configuration, league settings, team-name mappings, and model feature configuration |
| `train.py`                | Main command-line entry point                                                       |
| `requirements.txt`        | Lists the Python dependencies and their tested versions                             |
| `src/data_loader.py`      | Loads match data and performs season-based splitting                                |
| `src/elo.py`              | Calculates football seasons and Elo ratings                                         |
| `src/features.py`         | Creates rolling form and xG/xGA features                                            |
| `src/understat_loader.py` | Retrieves and prepares Understat data                                               |
| `src/training.py`         | Handles feature engineering, model training, prediction, and evaluation             |
| `football_data/`          | Contains local historical football match data                                       |
| `.gitignore`              | Specifies files and directories that should not be committed to the repository      |
| `README.md`               | Project documentation, setup instructions, methodology, and limitations             |

## Installation

### Requirements

The project requires Python and the dependencies listed in `requirements.txt`.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```


## Usage

The main entry point is:

```bash
python train.py --league PremierLeague
```

Other supported leagues are:

```bash
python train.py --league LaLiga
python train.py --league SerieA
```

The training pipeline then:

1. Loads the historical league data.
2. Retrieves the relevant Understat data.
3. Aligns the datasets.
4. Creates rolling features.
5. Calculates sequential Elo ratings.
6. Splits the data by season.
7. Trains the Random Forest and XGBoost models.
8. Generates validation predictions.
9. Calculates accuracy, log loss, and Brier score.
10. Outputs prediction probabilities.

---

## Reproducibility

For comparable results, the following should remain consistent:

* Historical CSV data
* Understat data
* Feature configuration
* Team-name mappings
* Elo parameters
* Random seeds
* Validation season
* Python/library versions

The current model configurations use:

```text
Random Forest random_state = 67
XGBoost random_state       = 67
```

The validation process is deterministic with respect to the same input data and configuration.

For stronger reproducibility in the future, dependency versions and experiment configurations should be pinned.

---

## Data Leakage Considerations

Avoiding future information is a central consideration in this project.

The current pipeline attempts to prevent common forms of leakage by:

* Keeping matches in chronological order
* Splitting validation by season
* Using only previous matches for rolling features
* Calculating Elo sequentially
* Using pre-match Elo values as model features
* Updating Elo only after processing the current result
* Holding out a later season rather than randomly sampling matches from the same period

For example, when predicting:

```text
Team A vs Team B
```

the model should only receive information that would have been available immediately before that fixture.

The project does not claim to be completely leakage-free. Dataset alignment, feature engineering, and future changes to the pipeline should continue to be reviewed for temporal leakage.

---

## Limitations

The current system has several limitations:

* Currently supports three leagues
* Feature coverage depends on the available historical datasets
* Uses a relatively small hand-engineered feature set
* Does not currently implement a general missing-value strategy
* Does not currently predict final scorelines
* Does not currently calibrate probabilities
* Uses a single latest-season validation split rather than full walk-forward validation
* Does not currently store formal experiment results
* Does not currently provide a fully automated live-data pipeline
* Elo parameters have not necessarily been optimized for every league

The model should therefore be considered an experimental football prediction system rather than a production forecasting system.

---

## Future Work

Planned improvements include:

### Poisson Goal Model

Implement a separate goal-based model to estimate expected goals for both teams and derive scoreline probabilities.

This would allow predictions such as:

```text
0-0: 8.2%
1-0: 14.7%
1-1: 12.5%
2-1: 10.3%
...
```

The resulting scoreline probabilities could then be converted into estimated probabilities for:

* Home win
* Draw
* Away win

### Model Comparison and Ensemble

Compare the classification models against the Poisson approach and investigate whether combining their predictions improves probability quality.

### Walk-Forward Validation

Evaluate the models over several historical seasons:

```text
Train: 2020/21 → 2022/23
Test:  2023/24

Train: 2020/21 → 2023/24
Test:  2024/25

Train: 2020/21 → 2024/25
Test:  2025/26
```

This would provide a more robust estimate of performance across different seasons.

### Probability Calibration

Investigate whether predicted probabilities require calibration using techniques such as:

* Platt scaling
* Isotonic regression

### Automated Data Updates

Automate the process of retrieving newly completed fixtures and updating the historical dataset.

---

## Disclaimer

This project is a football analytics and machine-learning experiment.

Predictions are probabilistic estimates and are not guaranteed outcomes. The project is intended for research, experimentation, and learning rather than as financial or gambling advice.

## License

No license is currently specified for this repository.
