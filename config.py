"""
Stores feature definitions for the evaluation model.
"""

feature_columns = [
    # Team points last five matches
    "HomePT5",  
    "AwayPT5",
    # Team goals scored last five matches
    "HomeGS5",
    "AwayGD5",
    # Team goals conceded last five matches
    "HomeGC5",
    "AwayGC5",
    # Team goal difference last five matches
    "HomeGD5",
    "AwayGD5"
]
