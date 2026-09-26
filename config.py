"""
Stores feature definitions for the evaluation model.
"""

# League settings.
LEAGUES = {
    "PremierLeague": {
        "football_data": "football_data/PremierLeague",
        "understat": "ENG-Premier League",
        "api_football_id": 39,
        "div": "E0",
        # Only the Premier League CSVs carry a Referee column, so the
        # augmenter writes it for this league only.
        "writes_referee": True,
    },
    "LaLiga": {
        "football_data": "football_data/LaLiga",
        "understat": "ESP-La Liga",
        "api_football_id": 140,
        "div": "SP1",
        "writes_referee": False,
    },
    "SerieA": {
        "football_data": "football_data/SerieA",
        "understat": "ITA-Serie A",
        "api_football_id": 135,
        "div": "I1",
        "writes_referee": False,
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

# Mapping of team names between the API-Football fixtures endpoint and the
# football-data.co.uk CSVs, keyed by league directory name.
#
# Only names that actually differ are listed. API-Football names that are
# already identical to the CSV spelling are intentionally omitted. Any
# API-Football name that is neither identical nor listed here is treated as an
# error by src/api_football.py, so that a new or renamed team can never be
# silently ingested as a separate club (which would split that team's Elo
# rating and rolling form in two).
#
# The API-Football spellings below were taken from live /fixtures responses,
# not guessed. Note that they are a mix of short and long forms: the Premier
# League uses "Sheffield Utd" and "Wolves" rather than the full club names.
API_FOOTBALL_TEAM_MAP = {
    "PremierLeague": {
        "Manchester City": "Man City",
        "Manchester United": "Man United",
        "Newcastle": "Newcastle",
        "Nottingham Forest": "Nott'm Forest",
        "Sheffield Utd": "Sheffield United",
        "Wolves": "Wolves",
    },
    "LaLiga": {
        "Athletic Club": "Ath Bilbao",
        "Atletico Madrid": "Ath Madrid",
        "Celta Vigo": "Celta",
        "Granada CF": "Granada",
        "Rayo Vallecano": "Vallecano",
        "Real Betis": "Betis",
        "Real Sociedad": "Sociedad",
    },
    "SerieA": {
        "AC Milan": "Milan",
        "AS Roma": "Roma",
        "Hellas Verona": "Verona",
        "Inter": "Inter",
    },
}

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
