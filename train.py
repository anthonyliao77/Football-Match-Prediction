"""
Trains, validates and evaluates the prediction model.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    # confusion_matrix,
    log_loss,
)
from sklearn.preprocessing import LabelEncoder

# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier

from config import (
    FEATURE_COLUMNS,
    LEAGUES,
    TRAIN_END_DATE,
    VALIDATION_END_DATE,
    VALIDATION_START_DATE,
)
from data_loader import load_data
from elo import create_elo_features
from features import create_features
from understat_loader import load_understat_data

# Choosen league to train and evaluate the model on
# LEAGUE = "SerieA"
LEAGUE = "PremierLeague"
# LEAGUE = "LaLiga"
LEAGUE_CONFIG = LEAGUES[LEAGUE]

# Load and combine match data for the specified league
dataframe = load_data(
    league=LEAGUE_CONFIG["football_data"]
)

# Load xg data from understat for the specified league and season range
xg_data = load_understat_data(
    league=LEAGUE_CONFIG["understat"],
    start_year=2020,
    end_year=2025
)

# Merge the match data and xg data based on date and team names
dataframe = pd.merge_asof(
    dataframe,
    xg_data,
    left_on="Date",
    right_on="date",
    left_by=["HomeTeam", "AwayTeam"],
    right_by=["home_team", "away_team"],
    direction="nearest",
    tolerance=pd.Timedelta(days=1)
)

# Create features for the model using the combined match data
dataframe = create_features(dataframe=dataframe)
dataframe = create_elo_features(dataframe=dataframe)

# Split the data into training and validation sets based on date ranges
train_data = dataframe[dataframe["Date"] <= TRAIN_END_DATE]
validation_data = dataframe[
    (dataframe["Date"] >= VALIDATION_START_DATE)
    & (dataframe["Date"] <= VALIDATION_END_DATE)
]

# training data
train_X = train_data[FEATURE_COLUMNS]
train_y = train_data["FTR"]

# validation data
val_X = validation_data[FEATURE_COLUMNS]
val_y = validation_data["FTR"]

# Encode target labels for XGBoost
# Convert A/D/H to 0/1/2 for multiclass classification
label_encoder = LabelEncoder()

train_y_encoded = label_encoder.fit_transform(train_y)
val_y_encoded = label_encoder.transform(val_y)


# Random Forest model
random_forest = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=67,
    n_jobs=-1
)

random_forest.fit(train_X, train_y)

# Random Forest prediction
prediction = random_forest.predict(val_X)
rf_probabilities = random_forest.predict_proba(val_X)

rf_results = validation_data[
    ["Date", "HomeTeam", "AwayTeam", "FTR"]
].copy()

rf_results["Prediction"] = prediction

# Add predicted probabilities to the results DataFrame
rf_results["AwayProbability"] = rf_probabilities[:, 0]
rf_results["DrawProbability"] = rf_probabilities[:, 1]
rf_results["HomeProbability"] = rf_probabilities[:, 2]

# # Random Forest accuracy
rf_accuracy = accuracy_score(val_y, prediction)

# Log loss
rf_log_loss = log_loss(
    val_y,
    rf_probabilities,
    labels=random_forest.classes_
)

# Multiclass Brier score
one_hot_y = pd.get_dummies(
    val_y
).reindex(
    columns=random_forest.classes_,
    fill_value=0
)

rf_brier_score = (
    (rf_probabilities - one_hot_y.to_numpy()) ** 2
).sum(axis=1).mean()

print(
    rf_results[
        [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTR",
            "Prediction",
            "HomeProbability",
            "DrawProbability",
            "AwayProbability",
        ]
    ].head(10)
)
print(
    "Random Forest accuracy:", rf_accuracy,
    "Log loss:", rf_log_loss,
    "Brier score:", rf_brier_score,
    "\n"
)

# # Random Forest confusion matrix
# print(random_forest.classes_)
# print(confusion_matrix(val_y, prediction))


# XGBoost model
xgb = XGBClassifier(
    n_estimators=200,
    random_state=67,
    learning_rate=0.01,
    n_jobs=-1
)

xgb.fit(train_X, train_y_encoded)

# XGBoost prediction
xgb_prediction_encoded = xgb.predict(val_X)
xgb_probabilities = xgb.predict_proba(val_X)

# Convert predictions back to A/D/H
xgb_prediction = label_encoder.inverse_transform(
    xgb_prediction_encoded
)

# XGBoost results
xgb_results = validation_data[
    ["Date", "HomeTeam", "AwayTeam", "FTR"]
].copy()

xgb_results["Prediction"] = xgb_prediction

# Add probabilities
xgb_results["AwayProbability"] = xgb_probabilities[:, 0]
xgb_results["DrawProbability"] = xgb_probabilities[:, 1]
xgb_results["HomeProbability"] = xgb_probabilities[:, 2]

# XGBoost accuracy
xgb_accuracy = accuracy_score(
    val_y,
    xgb_prediction
)

# Log loss
xgb_log_loss = log_loss(
    val_y_encoded,
    xgb_probabilities
)

# Multiclass Brier score
one_hot_y = pd.get_dummies(
    val_y_encoded
).reindex(
    columns=range(xgb_probabilities.shape[1]),
    fill_value=0
)

xgb_brier_score = (
    (xgb_probabilities - one_hot_y.to_numpy()) ** 2
).sum(axis=1).mean()

# Print results
print(
    xgb_results[
        [
            "Date",
            "HomeTeam",
            "AwayTeam",
            "FTR",
            "Prediction",
            "HomeProbability",
            "DrawProbability",
            "AwayProbability",
        ]
    ].head(10)
)

print(
    "XGBoost accuracy:", xgb_accuracy,
    "Log loss:", xgb_log_loss,
    "Brier score:", xgb_brier_score
)

# # XGBoost confusion matrix
# print(label_encoder.classes_)
# print(
#     confusion_matrix(
#         val_y,
#         xgb_prediction,
#         labels=label_encoder.classes_
#     )
# )


# # Match distribution of validation data
# print(validation_data["FTR"].value_counts(normalize=True))


# # Random Forest feature importance
# rf_importance = pd.Series(
#     random_forest.feature_importances_,
#     index=train_X.columns
# ).sort_values(ascending=False)

# print("Random Forest feature importance:")
# print(rf_importance)


# # XGBoost feature importance
# xgb_importance = pd.Series(
#     xgb.feature_importances_,
#     index=train_X.columns
# ).sort_values(ascending=False)

# print("XGBoost feature importance:")
# print(xgb_importance)
