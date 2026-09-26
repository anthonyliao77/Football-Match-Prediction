"""
Loads football goal data from understat.
"""

import pandas as pd

from config import TEAM_NAME_MAP
from src.understat_client import (
    UnderstatUnavailable,
    get_league_season,
)


class UnderstatDataError(RuntimeError):
    """Raised when Understat returns no usable schedule data."""


def _clean(xg_data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalises a frame of fixtures so it can be lined up with the local CSVs.

    Parameters:
        xg_data (pd.DataFrame): The fixtures read for one league.

    Returns:
        pd.DataFrame: The same fixtures, sorted by date, with the team names
        rewritten to the football-data.co.uk spelling.
    """
    xg_data = xg_data.copy()

    # Team names are translated first, because the mapping is written in terms of
    # the names Understat uses.
    for column in ("home_team", "away_team"):
        xg_data[column] = xg_data[column].replace(TEAM_NAME_MAP).astype(str)

    return xg_data.sort_values("date").reset_index(drop=True)


def load_understat_data(
    league: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Loads football goal data from understat.

    Each season is requested on its own rather than as one range. A range is
    answered as a single unit, so a season Understat has not published takes the
    whole request down with it and the missing data is only visible by comparing
    row counts. Asking season by season means an unavailable one is named in the
    output instead of quietly disappearing.

    Parameters:
        league (str): The Understat URL name of the league, e.g. "EPL".
        start_year (int): The starting year of the season range.
        end_year (int): The ending year of the season range.
    Returns:
        pd.DataFrame: A DataFrame containing the loaded data
        for the specified league.

    Raises:
        UnderstatDataError: If no season in the requested range has data.
    """
    requested = [
        f"{year}/{year + 1}"
        for year in range(start_year, end_year + 1)
    ]

    frames = []
    loaded = []
    unavailable = {}

    for season in requested:
        try:
            frames.append(get_league_season(league, season))
        except UnderstatUnavailable as error:
            unavailable[season] = str(error)
            continue

        loaded.append(season)

    if not frames:
        detail = "; ".join(
            f"{season}: {reason}" for season, reason in unavailable.items()
        )

        raise UnderstatDataError(
            f"Understat returned no data for {league} in any of the requested "
            f"seasons: {', '.join(requested)}. {detail}"
        )

    if unavailable:
        print(
            f"Understat has no data for {league} in "
            f"{', '.join(unavailable)}. Those seasons are absent from the "
            f"local CSVs' xG columns."
        )

    return _clean(pd.concat(frames, ignore_index=True))
