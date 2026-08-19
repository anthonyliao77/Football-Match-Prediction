"""
Loads football goal data from understat.
"""

import pandas as pd
import soccerdata as sd

from config import TEAM_NAME_MAP


def load_understat_data(
    league: str,
    start_year: int,
    end_year: int
) -> pd.DataFrame:
    '''
    Loads football goal data from understat.
        Parameters:
            league (str): The name of the football league to load data for.
            start_year (int): The starting year of the season range.
            end_year (int): The ending year of the season range.
        Returns:
            pd.DataFrame: A DataFrame containing the loaded data
            for the specified league.
    '''
    seasons = [
        f"{year}/{year + 1}"
        for year in range(start_year, end_year + 1)
    ]

    understat = sd.understat.Understat(
        leagues=league,
        seasons=seasons,
    )

    # Load the schedule data from understat
    xg_data = understat.read_schedule()

    # Normalize the date column to remove time information
    xg_data["date"] = pd.to_datetime(
        xg_data["date"]
    ).dt.normalize()
    xg_data = xg_data.sort_values("date")

    # Converting team names to match the football-data dataset
    xg_data["home_team"] = xg_data["home_team"].replace(TEAM_NAME_MAP)
    xg_data["away_team"] = xg_data["away_team"].replace(TEAM_NAME_MAP)

    # Converting team names to strings
    xg_data["home_team"] = xg_data["home_team"].astype(str)
    xg_data["away_team"] = xg_data["away_team"].astype(str)

    # Sort matches by date
    xg_data = xg_data.sort_values("date").reset_index(drop=True)

    return xg_data
