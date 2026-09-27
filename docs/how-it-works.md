<sub>Part of [Football Match Prediction](https://github.com/anthonyliao77/Football-Match-Prediction). Start at the [README](https://github.com/anthonyliao77/Football-Match-Prediction#readme).</sub>

# How it works

== Features


The models use historical information that should be available before the predicted fixture.

== Data Preprocessing


== Season-Based Validation


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

== Elo Rating System


The project implements a sequential Elo rating system in `src/elo.py`.

== Machine Learning Models


== Model Evaluation


The project evaluates both the predicted class and the quality of the predicted probabilities.
