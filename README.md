# Football Match Prediction

A football match prediction project using historical match statistics, rolling team-form features, Understat expected goals, and Elo ratings to predict **Home Win, Draw, or Away Win**.

**Models:** Random Forest · XGBoost  
**Leagues:** Premier League · La Liga · Serie A  
**Prediction type:** Three-class match outcome probabilities

> This is a research-oriented football analytics project focused on machine learning, feature engineering, evaluation, and experimentation. It is not a production betting or live prediction system.

## What it predicts

For each fixture, both models return the predicted outcome and three-class probabilities:

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

- **Date, Home, Away** — the fixture in question
- **Predicted outcome** — Home / Draw / Away (shown per model)
- **Probabilities** — Home, Draw, Away, summing to 100%

Run a single fixture or an interactive session (see [Quick start](#quick-start)).

## Quick start

### Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Training works entirely from the local CSV files and needs no credentials. API-Football is optional and documented in [API-Football (optional)](docs/api-football.md).

### Predict one fixture

```bash
python predict.py --league PremierLeague --home Arsenal --away Chelsea \
    --date 2026-12-06
```

Both models are shown by default because disagreements are often informative.

### Train and evaluate

```bash
python train.py --league PremierLeague
```

The xG is already stored in the CSVs, so training needs no network access. To refresh data from Understat, see [Keeping the data current](docs/data.md#keeping-the-data-current).

## How well it works

The held-out validation season is small by design (e.g., 40 matches for Premier League/Serie A, 59 for La Liga), so scores can move between runs for reasons that have nothing to do with the model. A difference of a few hundredths in log loss on those leagues is noise, not a finding. The models and evaluation choices are documented in [How it works](docs/how-it-works.md) and [Design notes](docs/design-notes.md#why-the-numbers-can-be-trusted).

## How it works

```text
Historical Match Data (CSV, xG already merged)
        │
        ▼
  Report xG Coverage
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

The data update side:

```text
Understat league list  ──►  sync_understat.py  ──►  full schedule + results + shots
Understat              ──►  refresh_data.py    ──►  runs the above + xG backfill
```

The core pipeline (features, Elo, season-split, models, evaluation) is covered in [How it works](docs/how-it-works.md). Data collection, sync, and the scheduled workflows are covered in [Data](docs/data.md).

## Where the data comes from

**[Football-Data](https://www.football-data.co.uk/)** — historical CSV files in `football_data/` provide the base match lists for each league and season. These CSVs are treated as the canonical season rows; Understat is used to augment them with schedule completeness, results, and xG where applicable.

**[Understat](https://understat.com/)** — provides the season schedule, final scores as matches finish, per-match shot counts, and expected goals (xG). Understat access uses a small cookie/Referer handshake; the refresh reads it directly and writes values back into the season CSVs (provenance recorded in `result_source`). See [Understat](docs/data.md#understat).

## Keeping the data current

The repository has two scheduled workflows that keep the committed CSVs in sync with Understat (see [The two scheduled workflows](docs/data.md#the-two-scheduled-workflows) for timings, permissions, and guard details). Locally, `python refresh_data.py` runs the sync and backfill for all leagues and reports what remains unwritten. The exit status is designed for both a pre-prediction gate (`--require-fresh`) and a tolerant monitor (`--require-fresh --max-age-days 7`).

> **Note on summer inactivity.** GitHub disables scheduled workflows after 60 days of repository inactivity. Football is quiet over the summer (May–August), so the workflows may be disabled when the new season starts. Re-enable them from the Actions tab when you return to active matchdays; the instructions for notifications and that re-enable are in [The two scheduled workflows](docs/data.md#the-two-scheduled-workflows).

## Project structure

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
├── docs/
│   ├── api-football.md
│   ├── data.md
│   ├── design-notes.md
│   └── how-it-works.md
├── .github/workflows/
│   ├── data-freshness.yaml
│   ├── data-refresh.yaml
│   └── tests.yaml
└── README.md
```

**Key entry points:** `train.py` (train/evaluate), `predict.py` (single fixture or interactive), `refresh_data.py` (sync + xG refresh with status report), `sync_understat.py` (schedule + results + shots), `backfill_xg.py` (xG backfill), `update_data.py` (optional API-Football augmentation).

## Limitations

- Supports three leagues (Premier League, La Liga, Serie A)
- Refits both models on every prediction run (no persisted model artifacts)
- Uses a single latest-season validation split rather than full walk-forward validation
- Does not predict final scorelines or calibrate probabilities
- Feature coverage depends on the available historical datasets
- Elo parameters have not been exhaustively optimized per league

For a fuller list and the context behind tradeoffs, see [Design notes](docs/design-notes.md) and [Limitations](docs/how-it-works.md#model-evaluation) where relevant.

## Documentation

- [How it works](docs/how-it-works.md) — features, preprocessing, validation, Elo, models, evaluation
- [Data](docs/data.md) — data sources, CSV structure, sync, workflows, merge rules
- [API-Football (optional)](docs/api-football.md) — credentials, commands, quotas, limits
- [Design notes](docs/design-notes.md) — why the numbers can be trusted, reproducibility, leakage, cost, future work

## Disclaimer

This project is a football analytics and machine-learning experiment. Predictions are probabilistic estimates and are not guaranteed outcomes. It is intended for research, experimentation, and learning rather than as financial or gambling advice.

## License

No license is currently specified for this repository.