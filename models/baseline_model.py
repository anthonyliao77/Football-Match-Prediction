"""
Creates and evaluates a football match prediction model using
historical match data from football-data.co.uk.
"""

import glob
import pandas as pd
from sklearn.linear_model import LogisticRegression

# import dataset
# choose football league
files_path = glob.glob("PremierLeague/EPL-25-26.csv")
files_list = []

# read dataset
for file in files_path:
    files_list.append(pd.read_csv(file))

# create dataframe
dataframe = pd.concat(files_list, ignore_index=True)
# print(dataframe)

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

# input(X) and prediction(y)
X = dataframe[cols]
y = dataframe['FTR']

# model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# prediction
prediction = model.predict(X)

# results
results = dataframe[['Date', 'HomeTeam', 'AwayTeam', 'FTR']].copy()
results['prediction'] = prediction
print(results.head(10))

# incorrect predictions
# prediction_wrong = results[results['FTR'] != results['prediction']]
# print(prediction_wrong)
