"""
Computing features for the evaluation model.
"""

import pandas as pd

def last5_points(team: str, date: pd.Timestamp, dataframe: pd.DataFrame) -> int:
    '''
    Function used to compute a team's perfomance in last five matches.
    
        Parameters:
            team (str): Name of the team.
            data (pandas.Timestamp): Date of the current match.
            dataframe (pandas.Dataframe): DataFrame containing all match data.

        Returns:
            points (int): Total points earned in the team's previous five matches. 
    '''

def last5_goal_scored(team: str, date: pd.Timestamp, dataframe: pd.DataFrame) -> int:
    ''' 
    Function used to compute a team's goal scored in the last five matches.
    
        Parameters:
            team (str): Name of the team.
            data (pandas.Timestamp): Date of the current match.
            dataframe (pandas.Dataframe): DataFrame containing all match data.

        Returns:
            goal_conceded (int): The total goals scored in the team's previous five matches.
    '''

def last5_goal_conceded(team: str, date: pd.Timestamp, dataframe: pd.DataFrame) -> int:
    ''' 
    Function used to compute the amount of goals the team has conceded in the last five matches.
    
        Parameters:
            team (str): Name of the team.
            data (pandas.Timestamp): Date of the current match.
            dataframe (pandas.Dataframe): DataFrame containing all match data.

        Returns:
            goal_conceded (int): The total goals conceded in the team's previous five matches.
    '''

def last5_goal_difference(goal_scored: int, goal_conceded: int) -> int:
    ''' 
    Function used to compute a team's goal difference in the last five matches.
    
        Parameters:
            goals_scored (int): Amount of goals the team has scored.
            goals_conceded (int): Amount of goals the team has conceded. 

        Returns:
            goal_difference (int): The goal difference in the team's previous five matches.
    '''
    return goal_scored - goal_conceded
