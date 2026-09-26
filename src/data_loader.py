"""
Loads and combines football match data from dataset.
"""

import glob
import os
import tempfile
from pathlib import Path

import pandas as pd

from src.elo import get_season


DEFAULT_LINE_TERMINATOR = "\n"


def read_csv_format(path) -> tuple[str, str]:
    """
    Detects the byte-level formatting of an existing CSV.

    The season CSVs come from football-data.co.uk, which ships some of them with
    a UTF-8 BOM and all of them with Windows line endings. Reading one into
    pandas and writing it back out through ``to_csv`` silently drops the BOM and
    rewrites every line ending, which turns a change to three columns into a
    rewrite of the whole file and buries the actual edit in the diff.

    The convention is read from the file rather than fixed in config because the
    CSVs are not consistent with each other, and normalising them here would
    bury a real change all over again.

    Parameters:
        path: An existing CSV path.

    Returns:
        tuple[str, str]: The encoding to write with, and the line terminator.
    """
    with open(path, "rb") as handle:
        head = handle.read(4096)

    encoding = "utf-8-sig" if head.startswith(b"\xef\xbb\xbf") else "utf-8"

    terminator = (
        "\r\n" if b"\r\n" in head else DEFAULT_LINE_TERMINATOR
    )

    return encoding, terminator


def write_csv_atomic(
    path,
    dataframe: pd.DataFrame,
    encoding: str | None = None,
    line_terminator: str | None = None,
) -> None:
    """
    Writes a DataFrame to a CSV without risking a half-written file.

    The data is written to a temporary file in the destination directory and
    then moved into place, so an interrupted write cannot truncate a season of
    data. The temporary file shares the destination filesystem, which is what
    makes the move atomic.

    The encoding and line terminator default to whatever the destination file
    already uses, so rewriting it does not churn the whole file. A destination
    that does not exist yet, which is a season being created rather than
    updated, gets plain UTF-8 and Unix line endings.

    Parameters:
        path: The destination CSV path.
        dataframe (pd.DataFrame): The rows to write.
        encoding (str | None): Text encoding, or None to match the destination.
        line_terminator (str | None): Row terminator, or None to match.
    """
    destination = Path(path)

    if encoding is None or line_terminator is None:
        if destination.exists():
            detected_encoding, detected_terminator = read_csv_format(destination)

            encoding = encoding or detected_encoding
            line_terminator = line_terminator or detected_terminator
        else:
            encoding = encoding or "utf-8"
            line_terminator = line_terminator or DEFAULT_LINE_TERMINATOR

    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w",
        dir=destination.parent,
        delete=False,
        suffix=".tmp",
        encoding=encoding,
        newline="",
    ) as handle:
        temporary_path = Path(handle.name)
        dataframe.to_csv(
            temporary_path,
            index=False,
            encoding=encoding,
            lineterminator=line_terminator,
        )

    os.replace(temporary_path, destination)


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

    # Sort matches chronologically, breaking ties on the teams involved.
    #
    # The teams are not there for football reasons. A league plays several
    # matches on the same day, and sort_values defaults to an unstable sort, so
    # the order of those matches would be whatever order glob.glob happened to
    # return the files in. That order comes from the filesystem and differs
    # between two checkouts of the same commit, which changes the order of the
    # rows the rolling features and the sequential Elo ratings are built from,
    # which changes the model, which changes the reported scores. Adding the
    # teams makes the order depend only on the data.
    order = [
        column
        for column in ("Date", "HomeTeam", "AwayTeam")
        if column in dataframe.columns
    ]

    dataframe = dataframe.sort_values(order).reset_index(drop=True)

    return dataframe


def get_season_range(dataframe: pd.DataFrame) -> tuple[int, int]:
    """
    Determines the first and last season present in a dataframe.

    Both returned years are season start years, matching the convention
    load_understat_data expects, so the range maps directly onto the seasons
    that need to be requested. Deriving it from the data is what keeps the
    Understat request in step with the CSVs, instead of a hardcoded range that
    falls further behind each season.

    Parameters:
        dataframe (pd.DataFrame): A DataFrame containing a Date column.

    Returns:
        tuple[int, int]: The start years of the first and last seasons present
        (e.g. (2020, 2023) for data covering 2020/2021 to 2023/2024).
    """
    seasons = sorted(
        {get_season(date) for date in dataframe["Date"]}
    )

    start_year = int(seasons[0].split("/")[0])
    end_year = int(seasons[-1].split("/")[0])

    return start_year, end_year


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
