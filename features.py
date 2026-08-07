"""
Computing features for the evaluation model.
"""
from collections import defaultdict
import pandas as pd

def create_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    Creates a new DataFrame containing pre-match features for
    each fixture based on each team's historical performance.

    Parameters:
        dataframe (pd.DataFrame): DataFrame containing historical match
            data for all teams, which includes match statistics and results.

    Returns:
        pd.DataFrame: A new DataFrame containing the original match data
        along with features such as recent form, goals scored,
        goals conceded and other historical statistics.
    '''

    team_stats = defaultdict(list)
    features = []

    for _, match in dataframe.iterrows():
        home_team = match["HomeTeam"]
        away_team = match["AwayTeam"]

        # Collect team data from last five matches
        home_history = team_stats[home_team][-5:]
        away_history = team_stats[away_team][-5:]

        # Calculate all new features
        home_last5_points = last5_points(home_history)
        away_last5_points = last5_points(away_history)

        home_last5_goals_scored = last5_goal_scored(home_history)
        away_last5_goals_scored = last5_goal_scored(away_history)

        home_last5_goals_conceded = last5_goal_conceded(home_history)
        away_last5_goals_conceded = last5_goal_conceded(away_history)

        home_last5_goal_diff = last5_goal_difference(
        home_last5_goals_scored,
        home_last5_goals_conceded
        )

        away_last5_goal_diff = last5_goal_difference(
        away_last5_goals_scored,
        away_last5_goals_conceded
        )

        # Add own features to new dataframe
        features.append({
            "Date": match["Date"],
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "HomePT5": home_last5_points,
            "AwayPT5": away_last5_points,
            "HomeGS5": home_last5_goals_scored,
            "AwayGS5": away_last5_goals_scored,
            "HomeGC5": home_last5_goals_conceded,
            "AwayGC5": away_last5_goals_conceded,
            "HomeGD5": home_last5_goal_diff,
            "AwayGD5": away_last5_goal_diff,
            "FTR": match["FTR"],
        })

        # store team stats
        team_stats[home_team].append({
        "points": calculate_points(match["FTR"], True),
        "goals_scored": match["FTHG"],
        "goals_conceded": match["FTAG"]
        })

        team_stats[away_team].append({
        "points": calculate_points(match["FTR"], False),
        "goals_scored": match["FTAG"],
        "goals_conceded": match["FTHG"]
        })

    return pd.DataFrame(features)

def calculate_points(result: str, home: bool) -> int:
    """
    Calculates points earned from a match result.

    Parameters:
        result (str): Match result (H, D, A).
        home (bool): Whether the team played at home.

    Returns:
        int: Points earned.
    """

    if result == "D":
        return 1

    if result == "H" and home:
        return 3

    if result == "A" and not home:
        return 3

    return 0

def last5_points(team_points: list) -> int:
    '''
    Function used to compute a team's perfomance in last five matches.
    
        Parameters:
            team_stats (list): List of team's historical data.

        Returns:
            int: Total points earned in the team's previous five matches. 
    '''

    return sum(match["points"] for match in team_points)

def last5_goal_scored(team_goals: list) -> int:
    ''' 
    Function used to compute a team's goal scored in the last five matches.
    
        Parameters:
            team_goals (list): List of team's historical data.

        Returns:
            int: The total goals scored in the team's previous five matches.
    '''
    return sum(match["goals_scored"] for match in team_goals)


def last5_goal_conceded(team_conceded: list) -> int:
    ''' 
    Function used to compute the amount of goals the team has conceded in the last five matches.
    
        Parameters:
            team_conceded (list): List of team's historical data.

        Returns:
            int: The total goals conceded in the team's previous five matches.
    '''
    return sum(match["goals_conceded"] for match in team_conceded)

def last5_goal_difference(goal_scored: int, goal_conceded: int) -> int:
    ''' 
    Function used to compute a team's goal difference in the last five matches.
    
        Parameters:
            goals_scored (int): Amount of goals the team has scored.
            goals_conceded (int): Amount of goals the team has conceded. 

        Returns:
            int: The goal difference in the team's previous five matches.
    '''
    return goal_scored - goal_conceded
