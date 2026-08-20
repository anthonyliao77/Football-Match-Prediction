"""
Calculates, adjusts and updates ELO ratings for football teams.
"""

from collections import defaultdict

import pandas as pd

HOME_ADVANTAGE = 100  # ELO points added to the home team rating
K_FACTOR = 32  # ELO rating adjustment factor
INITIAL_RATING = 1500  # Initial ELO rating for all teams
A_FACTOR = 0.75  # Factor to adjust ELO ratings at the start of a new season
NEW_SEASON_MONTH = 7  # Month when a new football season starts (July)


def create_elo_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and updates ELO ratings for football teams
    based on match results.

    Parameters:
        dataframe (pd.DataFrame): A DataFrame containing football
        match data with columns for home team, away team, and match result.

    Returns:
        pd.DataFrame: A DataFrame with additional columns for home and away
        ELO ratings before the match and updated ELO ratings after the match.
    """
    # Initialize ELO ratings for all teams
    elo_ratings = defaultdict(lambda: INITIAL_RATING)

    # Track the previous season
    previous_season = None

    # Create new columns for ELO ratings
    dataframe["HomeEloBefore"] = 0.0
    dataframe["AwayEloBefore"] = 0.0
    dataframe["HomeEloAfter"] = 0.0
    dataframe["AwayEloAfter"] = 0.0

    for index, match in dataframe.iterrows():
        home_team = match["HomeTeam"]
        away_team = match["AwayTeam"]
        result = match["FTR"]
        current_season = get_season(match["Date"])

        # Apply regression to the mean at the start of a new season
        if previous_season is not None and current_season != previous_season:
            for team in elo_ratings:
                elo_ratings[team] = new_season_elo_regression(
                    elo_ratings[team]
                )

        previous_season = current_season

        # Get current ELO ratings
        home_rating = elo_ratings[home_team]
        away_rating = elo_ratings[away_team]

        # Store ELO ratings before the match
        dataframe.at[index, "HomeEloBefore"] = home_rating
        dataframe.at[index, "AwayEloBefore"] = away_rating

        # Calculate expected scores
        expected_home_score = calculate_expected_score(
            home_rating + HOME_ADVANTAGE, away_rating
        )
        expected_away_score = calculate_expected_score(
            away_rating, home_rating + HOME_ADVANTAGE
        )

        # Determine actual results for both teams
        home_result = get_match_result(result, home=True)
        away_result = get_match_result(result, home=False)

        # Update ELO ratings based on match results
        new_home_rating = update_elo_ratings(
            home_rating, home_result, expected_home_score
        )
        new_away_rating = update_elo_ratings(
            away_rating, away_result, expected_away_score
        )

        # Store updated ELO ratings after the match
        dataframe.at[index, "HomeEloAfter"] = new_home_rating
        dataframe.at[index, "AwayEloAfter"] = new_away_rating

        # Update the ELO ratings dictionary for future matches
        elo_ratings[home_team] = new_home_rating
        elo_ratings[away_team] = new_away_rating

    return dataframe


def calculate_expected_score(home_rating: float, away_rating: float) -> float:
    """
    Calculates the expected score for the home team based on ELO ratings.

    Parameters:
        home_rating (float): The ELO rating of the home team.
        away_rating (float): The ELO rating of the away team.
    """
    return 1 / (1 + 10 ** ((away_rating - home_rating) / 400))


def get_match_result(result: str, home: bool) -> float:
    """
    Determines the match result for a team.

    Parameters:
        result (str): The overall match result
        ("H" for home win, "A" for away win, "D" for draw).
        home (bool): A boolean indicating whether the team is home or away.
    Returns:
        float: The result for the specified team
        (1 for win, 0.5 for draw, 0 for loss).
    """
    if result == "D":
        return 0.5

    if result == "H" and home:
        return 1

    if result == "A" and not home:
        return 1

    return 0


def update_elo_ratings(
    rating: float,
    result: float,
    expected_score: float
) -> float:
    """
    Updates the ELO rating for a team based on match result.

    Parameters:
        rating (float): The current ELO rating of the team.
        result (float): The match result for the team
        (1 for win, 0.5 for draw, 0 for loss).
        expected_score (float): The expected score for the team.

    Returns:
        float: The updated ELO rating for the team.
    """
    return rating + K_FACTOR * (result - expected_score)


def new_season_elo_regression(old_rating: float) -> float:
    """
    Applies regression to the mean for ELO ratings
    at the start of a new season.

    Parameters:
        old_rating (float): The ELO rating of the team
        at the end of the previous season.

    Returns:
        float: The adjusted ELO rating for the team
        at the start of the new season.
    """
    return A_FACTOR * old_rating + (1 - A_FACTOR) * INITIAL_RATING


def get_season(date: pd.Timestamp) -> str:
    """
    Determines the football season based on the date.

    Parameters:
        date (pd.Timestamp): The date of the match.

    Returns:
        str: The football season in the format "YYYY/YYYY+1".
    """
    if date.month >= NEW_SEASON_MONTH:
        return f"{date.year}/{date.year + 1}"

    return f"{date.year - 1}/{date.year}"
