"""
Loads and combines football match data from dataset.
"""

import glob

import pandas as pd

from src.elo import get_season


def load_data(league: str) -> pd.DataFrame:
    """
    Loads and combines football match data from dataset.

    Parameters:
        league (str): The name of the football league to load data for.
    Returns:
        pd.DataFrame: A DataFrame containing the raw combined match data for
        the specified league.
    """
    files_path = glob.glob(f"{league}/*.csv")

    # Read all CSV files
    files_list = []

    for file in files_path:
        files_list.append(pd.read_csv(file))

    # Combine all the dataframes into a single dataframe
    dataframe = pd.concat(files_list, ignore_index=True)

    # Normalize dates and remove time information
    dataframe["Date"] = pd.to_datetime(
        dataframe["Date"],
        dayfirst=True
    ).dt.normalize()

    # Convert team names to strings
    dataframe["HomeTeam"] = dataframe["HomeTeam"].astype(str)
    dataframe["AwayTeam"] = dataframe["AwayTeam"].astype(str)

    # Sort matches chronologically
    dataframe = dataframe.sort_values("Date").reset_index(drop=True)

    return dataframe


def split_by_season(
    dataframe: pd.DataFrame,
    validation_seasons: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits match data chronologically by football season.

    Parameters:
        dataframe: DataFrame containing match data.
        validation_seasons: Number of complete seasons used for validation.

    Returns:
        Tuple containing training and validation DataFrames.
    """
    dataframe = dataframe.sort_values("Date").reset_index(drop=True)

    dataframe["Season"] = dataframe["Date"].apply(get_season)

    seasons = dataframe["Season"].unique()

    split_index = len(seasons) - validation_seasons

    training_seasons = seasons[:split_index]
    validation_seasons_list = seasons[split_index:]

    train_data = dataframe[
        dataframe["Season"].isin(training_seasons)
    ].copy()

    validation_data = dataframe[
        dataframe["Season"].isin(validation_seasons_list)
    ].copy()

    return train_data, validation_data
