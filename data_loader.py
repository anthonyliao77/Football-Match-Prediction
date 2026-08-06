"""
Loads and combines football match data from dataset.
"""

import glob
import pandas as pd

# Choose football league
files_path = glob.glob("LaLiga/*.csv")

# Read dataset
files_list = []
for file in files_path:
    files_list.append(pd.read_csv(file))

# Create dataframe
dataframe = pd.concat(files_list, ignore_index=True)

# Convert date to timestamps
dataframe["Date"] = pd.to_datetime(dataframe["Date"], dayfirst=True)
dataframe = dataframe.sort_values("Date").reset_index(drop=True)
