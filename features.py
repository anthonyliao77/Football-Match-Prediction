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

def last5_goal_scored(team: str, date: pd.Timestamp, dataframe: pd.DataFrame) -> int:
    ''' 
    Function used to compute a team's goal scored in the last five matches.
    
        Parameters:
            team (str): Name of the team.
            data (pandas.Timestamp): Date of the current match.
            dataframe (pandas.Dataframe): DataFrame containing all match data.

        Returns:
            int: The total goals scored in the team's previous five matches.
    '''

def last5_goal_conceded(team: str, date: pd.Timestamp, dataframe: pd.DataFrame) -> int:
    ''' 
    Function used to compute the amount of goals the team has conceded in the last five matches.
    
        Parameters:
            team (str): Name of the team.
            data (pandas.Timestamp): Date of the current match.
            dataframe (pandas.Dataframe): DataFrame containing all match data.

        Returns:
            int: The total goals conceded in the team's previous five matches.
    '''

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
