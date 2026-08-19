"""
Stores feature definitions for the evaluation model.
"""

LEAGUES = {
    "PremierLeague": {
        "football_data": "Football-data/PremierLeague",
        "understat": "ENG-Premier League",
    },
    "LaLiga": {
        "football_data": "Football-data/LaLiga",
        "understat": "ESP-La Liga",
    },
    "SerieA": {
        "football_data": "Football-data/SerieA",
        "understat": "ITA-Serie A",
    },
}

TRAIN_END_DATE = "2025-05-31"
VALIDATION_START_DATE = "2025-08-15"
VALIDATION_END_DATE = "2026-06-01"

feature_columns = [
    # Team points last five matches
    "HomePT5",
    "AwayPT5",
    # Team goals scored last five matches
    "HomeGS5",
    "AwayGS5",
    # Team goals conceded last five matches
    "HomeGC5",
    "AwayGC5",
    # Team goal difference last five matches
    "HomeGD5",
    "AwayGD5",
    # Team shots on target last five matches
    "HomeSOT5",
    "AwaySOT5",
    # Team shots last five matches
    "HomeS5",
    "AwayS5",
    # Team shots conversion last five matches
    "HomeSC5",
    "AwaySC5",
]
