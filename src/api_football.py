"""
This module receives data from the API-Football API.
"""

import os

# from pathlib import Path
import pandas as pd

# import requests
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_FOOTBALL_KEY")

headers = {
    "x-apisports-key": API_KEY
}


def update_league_data(league_id: int, season: int) -> None:
    """
    Updates the league data for a given league and season.

    Parameters:
        league_id (int): The ID of the league.
        season (int): The season year (e.g., 2023).
    """


def fetch_league_schedule(league_id: int, season: int) -> dict:
    """
    Fetches the league schedule for a given league and season.

    Parameters:
        league_id (int): The ID of the league.
        season (int): The season year (e.g., 2023).

    Returns:
        dict: A dictionary containing the league schedule.
    """


def fetch_fixture_details(fixture_ids: list[int]) -> dict:
    """
    Fetches detailed statistics for a batch of fixtures.

    Parameters:
        fixture_ids (list[int]): A list of fixture IDs.

    Returns:
        dict: A dictionary containing detailed fixture statistics.
    """


def transform_fixtures(response: dict) -> pd.DataFrame:
    """
    Transforms the API response into a DataFrame.

    Parameters:
        response (dict): The API response containing fixture data.

    Returns:
        pd.DataFrame: A DataFrame containing fixture information.
    """


def update_csv_file(league: str, season: str, new_data: pd.DataFrame) -> None:
    """
    Updates the CSV file for a given league and season with new fixture data.

    Parameters:
        league (str): The name of the league.
        season (str): The season year (e.g., "2023").
        new_data (pd.DataFrame): A DataFrame containing new fixture data.
    """
