"""
Trains, validates and evaluates the prediction model.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from config import (
    TRAIN_END_DATE,
    VALIDATION_END_DATE,
    VALIDATION_START_DATE,
    feature_columns,
)
from data_loader import load_data
from features import create_features

# Choosen league to train and evaluate the model on
LEAGUE = "Football-data/SerieA"

# Load and combine match data for the specified league
dataframe = load_data(league=LEAGUE)
dataframe = create_features(dataframe=dataframe)

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
