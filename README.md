# Football Match Prediction

A football match prediction project using historical match statistics, rolling team-form features, Understat expected goals, and Elo ratings to predict **Home Win, Draw, or Away Win**.

**Models:** Random Forest · XGBoost
**Leagues:** Premier League · La Liga · Serie A
**Prediction type:** Three-class match outcome probabilities

> This is a research-oriented football analytics project focused on machine learning, feature engineering, evaluation, and experimentation. It is not a production betting or live prediction system.

## Overview

The project follows a chronological football prediction pipeline:

1. Historical match data is loaded from local CSV files, which already carry the Understat xG merged in.
2. The xG coverage is reported, and any season Understat has not published yet is called out before the models are trained.
3. Historical rolling team statistics are calculated using previous matches.
4. xG and xGA information is incorporated into rolling features.
5. Elo ratings are calculated sequentially so that each fixture receives the ratings available **before that match**.
6. Data is split chronologically by football season rather than randomly.
7. Random Forest and XGBoost classifiers are trained on historical seasons.
8. The models are evaluated on the latest held-out season.
9. Predictions include both the predicted outcome and probabilities for Home, Draw, and Away.

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
* [x] Fixture prediction command line
* [x] Interactive prediction session
* [x] Elo and form context alongside the predicted probabilities
* [x] Full-season schedule sync from Understat
* [x] Result backfill from Understat, existing values never overwritten
* [x] Shot counts from Understat's per-match endpoint
* [x] A refresh that reports what is still unwritten, with an exit status for schedules
* [x] Automated daily refresh, committed to `main` without a pull request
* [x] A separate read-only freshness check that no longer depends on the refresh

### Planned

* [ ] Poisson goal prediction model
* [ ] Scoreline probability predictions
* [ ] Walk-forward validation across multiple seasons
* [ ] Probability calibration
* [ ] Additional leagues and historical data
* [ ] Comparison between classification and goal-based prediction models

---

## Data Pipeline

Understat xG is written into the season CSVs ahead of time by
`backfill_xg.py`, so a training run reads one file per season and makes no
network call. The runtime pipeline is:

```text
Historical Match Data (CSV, xG already merged in by backfill_xg.py)
        │
        ▼
  Report xG Coverage  ──── rows with measured xG, before any features
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

The offline half of that, which is a separate command:

```text
Football-Data CSVs  +  Understat  ──►  backfill_xg.py  ──►  CSVs with xG
```

And the two commands that keep the CSVs current between seasons:

```text
Understat league list  ──►  sync_understat.py  ──►  full schedule + results + shots
Understat              ──►  refresh_data.py    ──►  runs both of the above
```

`refresh_data.py` is the one to run. It calls `sync_understat.py` and
`backfill_xg.py` for every league in turn, reports what each one found, and
returns an exit status a schedule can act on. The other two exist so that
either half can be run on its own while working out why the report is saying
what it is saying.

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

Historical expected-goals data is obtained from Understat by `src/understat_client.py`, which reads the JSON endpoint the Understat website itself calls. See [Understat](#understat).

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

No API key is required for these local files. They can optionally be augmented with API-Football fixtures, which does require a key (see [Updating Match Data](#updating-match-data)).

### Understat

Understat provides historical expected-goals information.

There is no public Understat API, so `src/understat_client.py` reads the same
JSON endpoint the website itself calls:

```text
https://understat.com/getLeagueData/{league}/{start_year}
```

where `{league}` is one of `EPL`, `La_liga` and `Serie_A`, and `{start_year}` is
the first year of the season, so 2026/2027 is requested as `2026`. The response
lists every fixture of that season with both xG values, which are then matched
against the local CSVs.

Two details are load-bearing:

* **The endpoint needs a session.** It answers 404 until the client has first
  visited the site and holds its cookies, so the client fetches the homepage
  once before the first request. A 404 caused by a missing cookie is
  indistinguishable from a season that does not exist, which is a very
  confusing way to lose a season.
* **One request per season.** The endpoint can also be asked for several
  seasons at once, but it is answered as a unit: if the newest season is not
  published, the whole response comes back short and the missing data is only
  visible by counting rows. Requesting each season separately means an
  unpublished season costs exactly that season.

Only fixtures the site marks as played are kept. A fixture that has not been
played carries `xG: {h: null, a: null}`, and a season with no played matches at
all is reported as unavailable rather than returned as an empty frame.

Requests are sequential with a small delay and there is no cache. That is
deliberate: a cached in-progress season would go stale and silently, which is
the one failure mode that would be hardest to notice.

#### Storing xG in the CSVs

Understat xG is written into the season CSVs once, by `backfill_xg.py`:

```bash
python backfill_xg.py --dry-run          # report what would change
python backfill_xg.py --league LaLiga    # write one league
```

Three columns are appended to each season file: `home_xg`, `away_xg` and
`xg_source`. The backfill only ever fills a **gap**. A value already in the file
is left exactly as it is, and `xg_source` names where the values came from
(`understat`). Rerunning it writes nothing.

The point of storing this is that the CSVs stay the single source of truth: a
season that Understat published is readable without a network call, and the
training run does not depend on Understat being reachable or on it still
listing an old season.

Matches are matched on **exact date plus both teams**, so a fixture that was
rescheduled keeps its gap until the two sources agree on the date. A handful of
rows are in that state permanently because a match was postponed and replayed
on a different day.

#### When a season has no xG yet

Understat publishes a season's xG only once matches have been played, and it
can lag the fixture list, so the newest rows of a league that is still being
played can carry no xG at all. Nothing is filled in for them: filling with
**0** would be actively harmful, because a zero says "this team created no
chances", and the four rolling xG features would then read zero for every row
of the validation set.

Instead the training run says so, before it builds any features:

```text
xG coverage: 2320 of 2320 rows carry measured xG
```

and when the season being validated on is not fully covered:

```text
**************************************************************************
WARNING: 12 of 12 rows in 2026/2027 have no measured xG.
Understat has not published that season, so HomeXG5, AwayXG5,
HomeXGA5 and AwayXGA5 will read 0 for every row of the validation
set. The scores below are therefore pessimistic, and the cause will
not be visible in the metrics themselves.
Try: python backfill_xg.py --league PremierLeague
************************************************************************
```

The check is per season and only the newest one is checked, because that is the
season the reported scores are computed on. Gaps in older seasons are counted
in the coverage line but do not warn: they cannot affect the current scores.

The message says *pessimistic* rather than *optimistic* because the model sees
no xG evidence for the teams it is predicting, and falls back on the rest of
its features. It is guessing with less information, and the error is in the
direction of the missing signal.

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

There is no general-purpose imputation system, and none is needed for the
football-data columns, which are complete for the seasons in use.

xG is the one exception and is handled deliberately rather than generically: a
value Understat has published is used as is, and a value it has not is left
missing. See [When a season has no xG yet](#when-a-season-has-no-xg-yet).

Two rules keep this honest:

* A missing xG is never filled in, because the only automatic fillers
  available (zero, or something regressed from shot counts) both put invented
  numbers into features whose whole purpose is to be measured.
* Every training run prints how many rows carry measured xG, and warns by name
  if the season being validated on is among the gaps, so the proportion is
  never a silent assumption.

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

The most recent season before the one being predicted is held out for
validation, and the season under prediction is used for neither.

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

$$
E = \frac{1}{1 + 10^{(R_{\text{opp}} - R_{\text{team}})/400}}
$$

For a home fixture, the home team's rating receives the configured home-advantage adjustment before calculating the expected score.

### Rating Update

After the match result is known, the rating is updated using:

$$
R' = R + K(S-E)
$$

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

$$
R_{\text{new}} =
A \times R_{\text{old}}
+
(1-A) \times R_{\text{initial}}
$$

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

$$
Accuracy =
\frac{\text{Correct Predictions}}
{\text{Total Predictions}}
$$

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

> **Read the validation numbers in proportion.** The held-out season is 40
> matches for the Premier League and SerieA and 59 for LaLiga, so the scores
> move around between runs for reasons that have nothing to do with the model.
> A change in log loss of a few hundredths in those leagues is noise, not a
> finding.

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

## Predicting a Fixture

`train.py` reports how the models did on a season they had not seen.
`predict.py` uses them on a fixture that has not been played yet.

### One fixture from the command line

```bash
python predict.py --league PremierLeague --home Arsenal --away Chelsea \
    --date 2026-12-06
```

```text
PremierLeague - Arsenal v Chelsea, Sun 06 Dec 2026
Data through 2026-09-14 (2320 matches, 30 teams)

  Random Forest  Home  63%  Draw  16%  Away  20%   ->  HOME
  XGBoost        Home  56%  Draw  20%  Away  24%   ->  HOME

  Elo               1713  vs     1535     (home advantage +100)
  Elo expectancy    0.83  vs     0.17     (ratings only, no draw model)
  Form, last 5   Arsenal           15   10.00    2.00   12.08
  Form, last 5   Chelsea             7   11.00   11.00    8.98
                    pts  scored  conceded      xG
```

| Flag | Meaning |
| --- | --- |
| `--league` | Required. `PremierLeague`, `LaLiga` or `SerieA` |
| `--home` / `--away` | Required. Team names as the season CSVs spell them |
| `--date` | Optional, `YYYY-MM-DD`. Defaults to the date these two clubs next meet on the schedule |
| `--model` | `both` (default), `rf` or `xgb` |
| `--explain` | Also print the full feature vector behind the prediction |

Team and league names are matched loosely, so `epl`, `Serie A` and `  arsenal `
all work. A name that cannot be resolved is refused with the closest spellings
in the data rather than guessed at.

Both models are shown side by side by default because they disagree often
enough to be worth seeing. When they do, that disagreement is information, and
collapsing it to one number would hide it.

### A session

With no arguments it asks for fixtures one at a time:

```bash
python predict.py
```

```text
Predict an unplayed fixture. Enter 'quit' at any prompt to stop.

League (PremierLeague / LaLiga / SerieA):
Loading PremierLeague and training... done.
Home team:
Away team:
Date (YYYY-MM-DD, blank for 2026-09-21):
```

A league is asked for every fixture, so a session can move between leagues. Its
models are built the first time that league is asked for and then kept, so
staying in one league pays for the training once.

`quit`, `exit` or `q` stops the session from any prompt, as does the end of the
input.

### What it refuses

* A team that appears in neither the results nor the season's schedule, listing
  the closest names.
* A team playing itself.
* A fixture whose result is already in the data, or a date the data has already
  reached.

A club that is on the schedule but has **not played yet** is not refused. Its
debut is the fixture most worth being able to ask about, and refusing it because
the club is new would refuse the prediction the data supports least and is
needed most. The prediction carries a note saying the club has no matches in the
data, so its Elo starts at 1500 and its form window is empty, and the model is
effectively working without that side.

When no date is given, the default is read from the schedule rather than
invented: the date those two clubs next meet, or the next fixture in the league
if they are not due to play. A fixed week after the last match was right most
weeks in midwinter and wrong across every international break, at the end of a
season, and any time a fixture was moved, which is the worst kind of default:
usually correct, and silently wrong the rest of the time.

Two clubs having met before is deliberately **not** a reason to refuse. Most
real fixtures are a repeat of a pairing from the same or a previous season, and
refusing those would rule out most of what anyone would want to ask about.

### Why the numbers can be trusted

The hard part is not fitting a model but making sure the feature vector a
prediction is scored against is the one the model was fitted on. Recomputing
features for a fixture by hand is where this normally goes wrong, so nothing is
recomputed.

Instead the fixture is appended to the data as a synthetic row with its result,
score and xG left empty, and the ordinary `create_features` and
`create_elo_features` pass runs over the result. The synthetic row's own features
are then read back by position.

```text
Real matches ──► append one synthetic row ──► the normal feature pass
                                                      │
                                     read that row's features by position
```

That is the same code path training uses, so the two cannot drift apart. The
rolling statistics naturally read the five matches before the fixture, and the
Elo rating is whatever it was immediately before kickoff, with new-season
regression applied if the date crosses the July boundary.

The test suite checks this directly: one match is deleted from the data, asked
for by name, and every one of its twenty features is asserted equal to the
vector the training pipeline produced for that row while it was still there. The
row is then also checked not to be the last row in the frame, because taking the
last row would be the natural shortcut and would return a different match's
features.

The result, score and xG columns are left empty rather than filled with zero on
purpose. A placeholder `0-0` would be indistinguishable from a real goalless
draw if the row's position ever shifted.

### Cost and honesty about it

There is no saved model. Every run loads the CSVs, rebuilds the features and
Elo ratings, and refits both classifiers on every season. Measured on the
Premier League data, that is about 1.8s to fit and about 0.5s per fixture
afterwards, so roughly 2.7s for a single command-line prediction. That is a
deliberate trade for a codebase with no artifact to go stale, and it is the
thing to revisit first if this ever needs to be quick.

The model is fitted on every season **before** the validation season, and the
season being predicted is in neither the fit nor the validation split. A season
after the validation season is not quietly folded into training: fitting on it
would train the model on the very matches it is about to predict, and the
reported scores would stop describing the model being used.

The rolling features and Elo ratings behind a prediction are a separate
question from the fit. Those are built from every match played so far,
including the season being predicted, because a result that has already happened
is real information about how a team is playing. A fixture is only ever a date
and two clubs, so a fixture that has not been played is dropped before the
feature and Elo passes: those passes are chronological and would read a row
with no result as a zero for both sides, spending a slot in every rolling window
that follows it.

Caveats travel with the prediction rather than being buried here. A team with a
thin history, a date older than the data, a season boundary and an unmeasured
xG window are each reported in the output, because a probability printed
without them invites reading more into it than it can carry.

---

## Project Structure

```text
Football-Match-Prediction/
├── config.py
├── train.py
├── predict.py
├── update_data.py
├── backfill_xg.py
├── sync_understat.py
├── refresh_data.py
├── requirements.txt
├── football_data/
│   ├── LaLiga/
│   ├── PremierLeague/
│   └── SerieA/
├── src/
│   ├── api_football.py
│   ├── data_loader.py
│   ├── elo.py
│   ├── features.py
│   ├── predict.py
│   ├── results.py
│   ├── training.py
│   ├── understat_client.py
│   ├── understat_loader.py
│   └── xg.py
├── .github/
│   └── workflows/
│       ├── data-freshness.yaml
│       ├── data-refresh.yaml
│       └── tests.yaml
├── .gitignore
└── README.md
```

### Main Files

| File                         | Purpose                                                                             |
| ---------------------------- | ----------------------------------------------------------------------------------- |
| `config.py`                  | Configuration, league settings, team-name mappings, and model feature configuration |
| `train.py`                   | Main command-line entry point for training                                          |
| `predict.py`                 | Command-line entry point for predicting a single fixture, and the interactive session |
| `update_data.py`             | Command-line entry point for augmenting the local CSVs with API-Football fixtures    |
| `backfill_xg.py`             | Command-line entry point for writing Understat xG into the season CSVs              |
| `sync_understat.py`          | Command-line entry point that adds the rest of a season's schedule and fills in results |
| `refresh_data.py`            | Runs the fixture and xG refresh for every league and reports what is still unwritten |
| `requirements.txt`           | Lists the Python dependencies and their tested versions                             |
| `src/api_football.py`        | Fetches API-Football fixtures and merges them into the local CSVs                   |
| `src/data_loader.py`         | Loads match data, performs season-based splitting, and writes CSVs atomically       |
| `src/elo.py`                 | Calculates football seasons and Elo ratings                                         |
| `src/features.py`            | Creates rolling form and xG/xGA features                                            |
| `src/predict.py`             | Fits the models on the training seasons and predicts an unplayed fixture            |
| `src/results.py`             | The result and shot count columns, and the provenance value written with them       |
| `src/understat_client.py`    | Reads Understat's per-season league endpoint and its per-match shots endpoint        |
| `src/understat_loader.py`    | Retrieves and prepares Understat data, one season at a time                          |
| `src/xg.py`                  | Defines the xG columns and the provenance value written with them                    |
| `src/training.py`            | Handles feature engineering, model training, prediction, and evaluation             |
| `tests/`                  | Test suite, run with `python -m pytest`                                             |
| `tests/factories.py`      | Builds the synthetic leagues the prediction tests run against                      |
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

### API-Football Credentials

Training works entirely from the local CSV files and needs no credentials. To
augment those files with API-Football data, create a `.env` file in the project
root:

```bash
API_FOOTBALL_KEY=your_api_key_here
```

The key is read only when an API request is made, so the project still imports
and trains normally without it. `.env` is listed in `.gitignore` and must not be
committed.

Get a key from [api-football.com](https://www.api-football.com/). The free plan
is enough to try the tooling, but it restricts which seasons can be queried and
does not allow the `ids` parameter, so match statistics cannot be fetched with
it. A paid plan removes both limits.


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

The xG is already stored in the CSVs, so training needs no network access. To
refresh it from Understat, for example after a weekend of matches:

```bash
python backfill_xg.py --dry-run    # report what would change first
python backfill_xg.py
```

The training pipeline then:

1. Loads the historical league data, including the Understat xG already stored in the CSVs.
2. Reports how much of the data carries measured xG, and warns if the validation season has gaps.
3. Creates rolling features.
4. Calculates sequential Elo ratings.
5. Splits the data by season.
6. Trains the Random Forest and XGBoost models.
7. Generates validation predictions.
8. Calculates accuracy, log loss, and Brier score.
9. Outputs prediction probabilities.

No network call is made during training, and no Understat data is merged at
runtime. Everything xG-related is read from the CSVs and topped up in memory.

---

## Updating Match Data

The CSVs in `football_data/` are the single source of truth for training. They
come from Football-Data.co.uk and can be augmented with fixtures from
API-Football using `update_data.py`.

### A season CSV holds the whole season, played and upcoming

Each season file holds that season's full fixture list, not just the results.
A fixture that has not been played yet is present with a date and two clubs,
and blank everywhere else: result, goals, shots and xG. This is what makes the
season being predicted known in advance, and it is why the whole file is 380
rows rather than the handful played so far.

Those blank rows are inert. Everything downstream filters on a result being
present, so an unplayed fixture never reaches the rolling form, the xG columns
or the Elo pass. `test_a_pending_fixture_in_the_csv_cannot_change_a_prediction`
holds that line: padding a season file with the rest of the schedule cannot move
a prediction.

Two consequences for anyone reading the files by hand:

* A blank `FTR` means **not yet played**, not a nil-nil draw. A nil-nil draw has
  `FTR` of `D`.
* Row count is no longer a proxy for how much data a season has. The number of
  played matches is.

### Adding the schedule and filling in results

`sync_understat.py` keeps a season CSV in step with Understat: it adds the
fixtures still to come, and it writes the results of matches that have since been
played. Understat lists a full season before kickoff, and publishes a score as
soon as a match ends, so one script covers both ends of a season's life:

```bash
python sync_understat.py --league PremierLeague --season 2026/2027
```

Use `--dry-run` first to see what would change, and `--season` to target a
specific season. The rules it works by:

* It **only adds or fills blanks**. A cell that already holds a value is left
  exactly as it is, so every result downloaded from football-data.co.uk is
  never overwritten. Understat fills the gaps the download has not reached yet,
  which is why the two sources coexist and neither one has to be re-downloaded.
* A match the two sources **disagree** on is left alone and reported. The value
  on file wins, and the difference is printed rather than resolved, because
  which of them is right is not something a script can decide.
* A row that already carries a result but is **missing shot counts** is left
  alone and reported, by the same reasoning: the goals on it came from
  football-data.co.uk and the shots would come from Understat, and mixing the two
  inside one match is worse than an honest gap. The all-or-nothing rule below
  protects the rows this script writes; this is the same corrupt shape arriving
  by another route, and it is named rather than tolerated.
* It matches on date, home team and away team, and translates Understat's club
  names through `TEAM_NAME_MAP` first.
* It writes only the columns the file already has, so a season that has never
  been given an xG column does not acquire an empty one.
* A club the season file has never seen is reported and left out, because two
  spellings of one club would split its Elo rating and its form in two.
* A **postponed** fixture has its date rewritten in place rather than added
  again. A league does play the same pairing more than once in a season, so only
  rows with no result are eligible to be re-dated; by the time the return leg is
  listed, the first one has been played and is left alone.
* A played match the file has never heard of is **added**, decided, with its
  shots. More often it is a fixture added before kickoff and still sitting there
  blank.

#### A result is never written without its shots

This is the rule the whole result-filling half turns on. Understat's league
payload carries goals but not shots; the shot counts live in a separate
per-match endpoint. So filling a result takes one request per match filled, and
those requests are spent **only** on matches that are actually being written —
a season already on file costs none.

If those counts cannot be read, the match is left blank, **score included**. That
looks like throwing away a result that is sitting right there, and it is
deliberate: `src/features.py` coerces a missing number to zero on purpose, so a
row given a score and no shots is not a row with blanks in it. It is a row
claiming the club managed no shots, the rolling features would sum it happily,
and the answer would be quietly wrong by about twenty shots with nothing
anywhere saying so. A blank row is honest and gets dropped; a plausible wrong
row gets trained on. A 0-0 where neither side recorded a shot is also left
blank, since that is what an abandoned or unplayed fixture looks like. Real
0-0 draws have shots in them and are written normally.

Rows this script fills are marked in a `result_source` column, so a fetched
value is told apart from a downloaded one. Historical rows are left unmarked on
purpose: a column claiming football-data.co.uk on every row of every season
would be a column nobody could trust to mean anything.

#### The two sources do not agree exactly, and that is fine

Measured over the 31 played Premier League 2026/2027 matches, comparing what
Understat reports against the football-data.co.uk CSV already on file:

| Column | Exact | Typical gap |
| --- | --- | --- |
| `HS` / `AS` total shots | 59 of 62 sides | 1 |
| `HST` / `AST` shots on target | 45 of 62 sides | 1 |

Almost every disagreement is a single shot, with one side out by two. The shot
vocabulary is fully accounted for, so this is two vendors counting the same
event slightly differently rather than a parse losing a category. Goals
disagree nowhere at all.

That gap is the reason the precedence rule is "existing values always win" and
not "Understat is more accurate". It also has a consequence worth stating
plainly: once a match is filled from Understat, the rolling shot features
average rows from two vendors, and a five-match form window can contain a ±1
difference that no amount of care at the boundary would remove. The feature is a
form signal and not a shot audit, so the noise is well inside its tolerance, but
it is a real seam in the data and the `result_source` column exists so it can be
located rather than remembered.

Running it twice changes nothing, which is what makes it safe to put on a
schedule.

### Keeping the data current

```bash
python refresh_data.py --dry-run   # report
python refresh_data.py             # write
```

This runs both Understat-backed steps for every league, fixtures then xG, and
prints one table:

```text
League            Fixtures  Re-dated  Results      xG  Unwritten
----------------------------------------------------------------
PremierLeague            0         0        0       0         0
LaLiga                   0         0        0       0         0
SerieA                   0         0        0       0         0
```

`--league` narrows it to one league and `--season` to one season of fixtures.
The xG step always covers the whole league, since it only fills cells that are
empty and narrowing it would hide gaps in older seasons.

An **Unwritten** count is the number of played matches the CSV has no result for.
That is the one figure worth reading, and it is the reason the command exists.
It is the only place in the project that can say a played match is missing,
because a fixture added before kickoff sits in the file with a blank result,
matches on date and sides, and is counted as *already present* by both
underlying scripts. Nothing else says so. Meanwhile a blank row reads as a match
that has not happened, so it is dropped from the rolling features and from the
Elo, and a club's recent form goes stale quietly.

**Results come from Understat too.** A blank result is no longer a chore left
undone, and the refresh no longer tells you to download anything. Understat
publishes a score as soon as a match ends, so the file fills itself in now, and
a non-zero Unwritten count means the refresh could not *finish*: an abandoned or
unplayed fixture, a match whose shot counts could not be read, or a request that
did not come back. Each one is listed by name by the fixtures step.

The exit status is non-zero if any league could not be refreshed, and a league
that fails does not stop the others. Understat publishes no API and no
availability, so treat a failure as retryable rather than as a broken dataset.

A league that failed shows a **dash** in the Unwritten column rather than a
zero, because the refresh died before it could count. A dash is not a zero, and
the run says so instead of reporting the leagues it did manage to check as proof
that everything is current.

### Before a prediction

```bash
python refresh_data.py --require-fresh
```

The default exit status catches a league failing, which is the loud failure. It
does not catch the likelier one: both Understat steps succeed, the tables are
clean, and one match is still sitting unwritten, so the run is green and the
model trains on a season that is quietly short a game. `--require-fresh`
extends the non-zero exit to a non-zero Unwritten count, and to any league whose
count is unknown.

Gate a prediction on it, and chain the two so the prediction cannot run on a
season that is short a game:

```cron
17 7 * * *  cd /path/to/repo && .venv/bin/python refresh_data.py --require-fresh \
             && .venv/bin/python predict.py --league PremierLeague --home Arsenal --away Leeds
```

A `--require-fresh` run will start failing the moment a matchday ends and the
next refresh has not yet run. That is the correct behaviour here: it is telling
you the model is about to be trained on results that are not there yet.

### As a monitor

The same command is the wrong guard for a check that runs on a timer, for a
reason that has nothing to do with correctness. Committed data goes stale after
every matchday and only becomes current once the refresh is run and its output
committed. Asked daily, a strict check is red most of the time, and a red check
people have learned to ignore is worse than no check.

`--max-age-days N` narrows the failure to a match that has been sitting
unwritten for more than N days:

```bash
python refresh_data.py --dry-run --require-fresh --max-age-days 7
```

A match blank since this morning is one nobody has got to yet. One blank for
three weeks is a fault nobody has looked at. The table shows both counts, so a
clean exit next to a non-zero Unwritten is visible rather than confusing:

```text
League            Fixtures  Re-dated  Results      xG  Unwritten  Overdue
----------------------------------------------------------------
PremierLeague            0         0        0       0          3        2
```

Two things it does not excuse. Without the flag the behaviour is exactly as it
was, since `0` would mean "older than today" and would quietly stop counting a
match played a few hours ago. And a league that *failed* still fails either way,
because a league nobody could count is not a young match — the threshold filters
match ages, and it cannot turn unknown into fine.

Ages are measured against UTC, so a local run and a scheduled one agree about
what "three days old" means.

### The two scheduled workflows

Both run daily and both can be started by hand from the Actions tab.

| Workflow               | Time (UTC) | Permission       | What it does                                                       |
| ---------------------- | ---------- | ---------------- | ------------------------------------------------------------------ |
| `data-freshness.yaml`  | 07:17      | `contents: read` | Dry-runs the refresh with a 7-day grace period, and fails if a match has been unwritten longer than that |
| `data-refresh.yaml`    | 08:17      | `contents: write`| Refreshes for real and commits the diff to `main`                  |

The check runs first and reads yesterday's settled tree. What it is for is the
failure the refresh cannot fix by itself: a match abandoned at Understat, a
league unreachable for a week, a push rejected because the branch moved. Those
are the cases where data is still unwritten the next morning despite a refresh
having run, and a check that only ever agreed with the writer would never notice
them.

**The refresh commits to `main` directly, with no pull request.** The diff is
only ever season CSVs, and the values in them are the same Understat figures the
project already trusts, so there is nothing in it for a human to decide. Every
committed value carries its origin in the `result_source` column, so a row that
was not on file before is still traceable after it lands.

`data-refresh.yaml` holds `contents: write`, which is the one genuinely widened
permission here, and it is what the guard is for: if anything other than a
season CSV is ever staged, the job refuses to commit and reports what it found.
The refresh cannot write code into the repository, so the realistic worst case
is bad numbers in `football_data/` rather than a backdoor.

A league that cannot be read does not stop the job at the point it fails. Each
league is written all-or-nothing, so the ones that did read cleanly are still
committed, and the job goes red afterwards so the failure is not mistaken for a
quiet day. The push is neither forced nor rebased: a rejected push means someone
landed on `main` mid-run, and the next run reconciles against it rather than
overwriting it.

#### Two things to know about this running unattended

**Turn on failure emails.** This is a personal setting, not a repository one:
repository *Settings → Actions* has no notification option at all, which is the
easy place to go looking. The control is at
<https://github.com/settings/notifications>, under **System** → **Actions**, where
the dropdown reads "Don't notify" until changed. Set it to **Email** and tick
**Only notify for failed workflows**.

It does nothing unless you are also **watching** the repository — GitHub only
delivers workflow notifications for repos you watch, so the setting can look
correct and still stay silent. The account's email address has to be verified
too, since that is where the notification goes.

Without this, a red run is a page nobody opens, and the check is only worth what
its failures are worth. No code is involved.

**GitHub disables scheduled workflows after 60 days of repository inactivity,**
and this is a public repository, so the rule applies. Football is quiet over the
summer — the three leagues finish in May and resume in August, which is roughly
10 to 12 weeks, and the refresh will find nothing to commit for most of it. When
the new season starts, both workflows will be sitting disabled and the data will
silently stop updating. Re-enable them from the Actions tab, and they will run
on their own again. Re-enabling also re-points notifications, since a scheduled
workflow's runs are reported to whoever re-enabled it, so confirm the email
above is still the address you want before doing it. This is deliberate: a
keep-alive commit whose only purpose is to defeat an inactivity timer is more
machinery than two clicks in August.

The scheduled runs are free. GitHub-hosted runners are not billed for public
repositories at all, on any plan — the monthly minute allowances apply to
private repositories only.

### How the merge behaves

API rows are matched to existing CSV rows on **date, home team and away team**:

* A **matched** row has only its *empty* cells filled in. Existing
  football-data.co.uk values always win, so scores and the odds and handicap
  columns are never overwritten.
* An **unmatched** row is appended, but only if it has a final score.
* Running the same update twice leaves the file unchanged.

Team names differ between the two sources, so they are translated through
`API_FOOTBALL_TEAM_MAP` in `config.py`. An API team name that is neither
identical to the CSV spelling nor listed there is **rejected** with an error,
because ingesting it would file the club under a second name and split its Elo
rating and rolling form in two. When a team is promoted or renamed, add the
mapping before running an update.

`Referee` is written for the Premier League only, which is the only set of
CSVs that has the column. The value is not read anywhere in the model.

### Commands

Preview a change without writing anything:

```bash
python update_data.py --league PremierLeague --season 2026 --dry-run
```

Note that `--dry-run` still makes the API calls, so it does spend quota. It
suppresses the file write, not the request.

Fetch recently completed fixtures, restricted to a date window to keep the
request count down:

```bash
python update_data.py --league PremierLeague --season 2026 \
    --from 2026-09-19 --to 2026-09-26
```

Backfill a whole season from the fixture schedule:

```bash
python update_data.py --league PremierLeague --season 2026 --schedule
```

### Request cost

A daily window covering a handful of matches costs 1 request for the fixtures
plus 1 per 20 matches for statistics. A full season with no window costs about
20. Restricting `--from` and `--to` is the single biggest lever on quota usage.

### Statistics availability

The `/fixtures` endpoint does not return match statistics inline, so shots and
shots on target are fetched in a second call that passes fixture IDs. Plans
without access to the `ids` parameter cannot make that call. When it is
rejected, the update still ingests scores and results, prints a warning, and
leaves the shots columns untouched.

### What a free API-Football key can and cannot do

The key this project was developed against is on the Free plan, and that plan
is genuinely limited. Measured against it, not assumed:

| Capability | Free plan |
| --- | --- |
| Request rate | 10 requests per minute |
| Daily requests | 100 |
| Seasons reachable | 2022/2023 to 2024/2025 |
| Current 2026/2027 season | No |
| Batched `ids` lookups | No |

Consequences worth planning around:

* **The current season cannot be ingested via the API at all.** A 2026/2027
  update returns nothing useful on this plan, so current-season data comes from
  the football-data.co.uk CSVs and `update_data.py` is useful for the older
  seasons the plan can still reach.
* **The statistics call fails**, because it needs the `ids` parameter. This is
  why shots come from football-data rather than the API, and why xG comes from
  Understat directly rather than from the API.
* **`--dry-run` still spends quota.** It suppresses the write, not the request.

Anyone with a paid key gets working current-season updates, and API-Football
also serves an `expected_goals` statistic that could serve as a second opinion
on xG. That path is **not** implemented: it cannot be tested against the free
key, and shipping an unvalidated xG source would be worse than the measured
Understat values the project already has.


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
* Refits both models on every prediction run, because no model is saved to disk
* Fits the prediction models on all seasons, so the held-out scores reported by
  `train.py` do not describe the predicting model exactly
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

Understat fetching and merging is implemented in `src/understat_client.py` and
`src/api_football.py`, driven by `sync_understat.py` and `backfill_xg.py`. It
runs unattended: `data-refresh.yaml` fills the season CSVs from Understat every
morning and commits the diff, and `data-freshness.yaml` independently checks that
nothing has been left unwritten for a week. What is still not automated is
anything involving API-Football, whose key is a personal credential and whose
fixtures are a subset of what Understat already provides.


---

## Disclaimer

This project is a football analytics and machine-learning experiment.

Predictions are probabilistic estimates and are not guaranteed outcomes. The project is intended for research, experimentation, and learning rather than as financial or gambling advice.

## License

No license is currently specified for this repository.
