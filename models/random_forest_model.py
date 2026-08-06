"""Glob is used for specific search paths"""
import glob
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# import dataset
# choose football league
files_path = glob.glob("SerieA/*.csv")
files_list = []

# read dataset
for file in files_path:
    files_list.append(pd.read_csv(file))

# create dataframe
dataframe = pd.concat(files_list, ignore_index=True)

# Train 20/21 to 23/24 season
# Validate 24/25 season
# Predict 25/26 season

dataframe["Date"] = pd.to_datetime(dataframe["Date"], dayfirst=True)
train_data = dataframe[dataframe["Date"]<= "31-05-2025"]
validation_data = dataframe[(dataframe["Date"] >= "10-08-2025") & (dataframe["Date"] <= "31-05-26")]

# select important cols in dataset
cols = [
'HS',
'AS',
'HST',
'AST',
'HF',
'AF',
'HC',
'AC',
'HY',
'AY',
'HR',
'AR',
]

# training data
train_X = train_data[cols]
train_y = train_data["FTR"]

# validation data
val_X = validation_data[cols]
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
results['prediction'] = prediction
print(results.head(10))

# accuracy recording
accuracy = accuracy_score(val_y, prediction)
print(accuracy)
