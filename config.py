"""
Stores feature definitions for the evaluation model.
"""

LEAGUES = {
    "PremierLeague": {
        "football_data": "football_data/PremierLeague",
        "understat": "ENG-Premier League",
    },
    "LaLiga": {
        "football_data": "football_data/LaLiga",
        "understat": "ESP-La Liga",
    },
    "SerieA": {
        "football_data": "football_data/SerieA",
        "understat": "ITA-Serie A",
    },
}

# Mapping of team names between football-data and understat datasets
TEAM_NAME_MAP = {
    # Serie A
    "AC Milan": "Milan",
    "Parma Calcio 1913": "Parma",

    # Premier League
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "West Bromwich Albion": "West Brom",
    "Wolverhampton Wanderers": "Wolves",

    # La Liga
    "Athletic Club": "Ath Bilbao",
    "Atletico Madrid": "Ath Madrid",
    "Real Betis": "Betis",
    "Celta Vigo": "Celta",
    "Espanyol": "Espanol",
    "SD Huesca": "Huesca",
    "Real Oviedo": "Oviedo",
    "Real Sociedad": "Sociedad",
    "Real Valladolid": "Valladolid",
    "Rayo Vallecano": "Vallecano",
}

TRAIN_END_DATE = "2025-05-31"
VALIDATION_START_DATE = "2025-08-15"
VALIDATION_END_DATE = "2026-06-01"

FEATURE_COLUMNS = [
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
    # Team xG last five matches
    "HomeXG5",
    "AwayXG5",
    # Team xGA last five matches
    "HomeXGA5",
    "AwayXGA5",
    # Team ELO rating before the match
    "HomeEloBefore",
    "AwayEloBefore",
]
