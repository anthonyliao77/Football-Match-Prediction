"""
Trains, validates and evaluates the prediction model.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from config import (
    LEAGUES,
    TRAIN_END_DATE,
    VALIDATION_END_DATE,
    VALIDATION_START_DATE,
    feature_columns,
)
from data_loader import load_data
from features import create_features
from understat_loader import load_understat_data

# Choosen league to train and evaluate the model on
LEAGUE = "LaLiga"
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

# Split the data into training and validation sets based on date ranges
train_data = dataframe[dataframe["Date"] <= TRAIN_END_DATE]
validation_data = dataframe[
    (dataframe["Date"] >= VALIDATION_START_DATE)
    & (dataframe["Date"] <= VALIDATION_END_DATE)
]

# training data
train_X = train_data[feature_columns]
train_y = train_data["FTR"]

# validation data
val_X = validation_data[feature_columns]
val_y = validation_data["FTR"]

# pokemon training of data
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=67,
    n_jobs=-1
)
model.fit(train_X, train_y)

# magic prediction
prediction = model.predict(val_X)
results = validation_data[['Date', 'HomeTeam', 'AwayTeam', 'FTR']].copy()
results['Prediction'] = prediction
print(results.head(10))

# Testing accuracy, confusion matrix and feature importance
accuracy = accuracy_score(val_y, prediction)
print(accuracy)

# Confusion matrix
print(model.classes_)
print(confusion_matrix(val_y, prediction))

# Match distribution of validation data
print(validation_data["FTR"].value_counts(normalize=True))

# Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=train_X.columns
).sort_values(ascending=False)
print(importance)
