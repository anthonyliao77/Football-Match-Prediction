"""
Loads and combines football match data from dataset.
"""

import glob

import pandas as pd


def load_data(league: str) -> pd.DataFrame:
    '''
    Loads and combines football match data from dataset.

    Parameters:
        league (str): The name of the football league to load data for.
    Returns:
        pd.DataFrame: A DataFrame containing the raw combined match data for
        the specified league.
    '''
    files_path = glob.glob(f"{league}/*.csv")

    # Read dataset
    files_list = []
    for file in files_path:
        files_list.append(pd.read_csv(file))
    # Combine all the dataframes into a single dataframe
    dataframe = pd.concat(files_list, ignore_index=True)
    dataframe["Date"] = pd.to_datetime(dataframe["Date"], dayfirst=True)
    dataframe = dataframe.sort_values("Date").reset_index(drop=True)
    # Normalize the date column to remove time information
    dataframe["Date"] = pd.to_datetime(
        dataframe["Date"],
        dayfirst=True
    ).dt.normalize()

    return dataframe
