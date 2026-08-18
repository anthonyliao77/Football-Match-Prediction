"""
Loads and combines football match data from dataset.
"""

import glob

import pandas as pd

from features import create_features


def load_data(league: str) -> pd.DataFrame:
    '''
    Loads and combines football match data from dataset.

    Parameters:
        league (str): The name of the football league to load data for.
    Returns:
        pd.DataFrame: A DataFrame containing the combined match data for
        the specified league.
    '''
    files_path = glob.glob(f"{league}/*.csv")

    files_list = []
    for file in files_path:
        files_list.append(pd.read_csv(file))

    dataframe = pd.concat(files_list, ignore_index=True)
    dataframe["Date"] = pd.to_datetime(dataframe["Date"], dayfirst=True)
    dataframe = dataframe.sort_values("Date").reset_index(drop=True)

    # Add new features to dataframe
    dataframe = create_features(dataframe=dataframe)

    return dataframe
