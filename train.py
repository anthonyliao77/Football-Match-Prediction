"""
Trains, validates and evaluates the prediction model.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from config import (
    TRAIN_END_DATE,
    VALIDATION_END_DATE,
    VALIDATION_START_DATE,
    feature_columns,
)
from data_loader import load_data
from features import create_features

# Choosen league to train and evaluate the model on
LEAGUE = "LaLiga"

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
    random_state=42,
    n_estimators=200
    )
model.fit(train_X, train_y)

# magic prediction
prediction = model.predict(val_X)
results = validation_data[['Date', 'HomeTeam', 'AwayTeam', 'FTR']].copy()
results['Prediction'] = prediction
print(results.head(10))

# accuracy recording
accuracy = accuracy_score(val_y, prediction)
print(accuracy)
# Match distribution of validation data
print(validation_data["FTR"].value_counts(normalize=True))
