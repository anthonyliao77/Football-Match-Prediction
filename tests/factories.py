"""
Builders for synthetic football data, shared by the tests that need some.

The prediction and command line tests both need a league on disk that behaves
like a real one: several teams, several dates, and a mix of played and unplayed
fixtures. Keeping the builders here rather than in either test file means the
schedule is described once, and the two files cannot drift into disagreeing
about what the data looks like.
"""

import pandas as pd

TEAMS = [
    "Arsenal",
    "Chelsea",
    "Liverpool",
    "Man City",
    "Everton",
    "Fulham",
    "Tottenham",
    "Wolves",
]


def round_robin_pairs(teams):
    """
    Every ordered pairing of the teams, home leg first.

    Built with the circle method so that each run of ``len(teams) / 2``
    consecutive pairs is a perfect matching: every club plays exactly once.
    That is what lets a fixture be pulled out of the data and asked for by name
    without its own teams picking up a same-day match they should not have had,
    and it leaves unplayed pairings behind for the tests that need one.

    Parameters:
        teams (list): The club names to schedule.

    Returns:
        list: (home, away) tuples, no pairing repeated.
    """
    size = len(teams)

    pairs = []

    for leg in (0, 1):
        rotation = list(teams)

        for _ in range(size - 1):
            fixtures = [
                (rotation[index], rotation[size - 1 - index])
                for index in range(size // 2)
            ]

            if leg:
                fixtures = [(away, home) for home, away in fixtures]

            pairs += fixtures

            rotation = [rotation[0], rotation[-1]] + rotation[1:-1]

    return pairs


def build_rows(count=40, per_date=4, start="2024-08-10", with_xg=True):
    """
    Builds a deterministic run of matches over several teams.

    Only part of the round robin is played, so unplayed fixtures exist. Every
    date carries more than one match, so a fixture asked for on a date that
    already has fixtures does not land last in the frame.

    Parameters:
        count (int): How many matches to build.
        per_date (int): How many matches share a date.
        start (str): The date of the first round.
        with_xg (bool): Whether to include expected goals.

    Returns:
        list: One dict per match, in DD/MM/YYYY date order.
    """
    pairs = round_robin_pairs(TEAMS)[:count]

    dates = pd.date_range(start, periods=-(-count // per_date), freq="7D")

    records = []

    for index, (home, away) in enumerate(pairs):
        round_index = index // per_date

        home_goals = (index * 3) % 4
        away_goals = (index * 5) % 3

        if home_goals > away_goals:
            result = "H"
        elif home_goals == away_goals:
            result = "D"
        else:
            result = "A"

        records.append({
            "Date": dates[index // per_date].strftime("%d/%m/%Y"),
            "Div": "E0",
            "HomeTeam": home,
            "AwayTeam": away,
            "FTHG": home_goals,
            "FTAG": away_goals,
            "FTR": result,
            "HS": 8 + round_index % 6,
            "AS": 6 + round_index % 5,
            "HST": 3 + round_index % 4,
            "AST": 2 + round_index % 3,
            "home_xg": 0.6 + (index % 5) * 0.3,
            "away_xg": 0.5 + (index % 4) * 0.25,
        })

    if not with_xg:
        for record in records:
            record["home_xg"] = None
            record["away_xg"] = None

    return records


def unplayed_pair(rows, extra=()):
    """
    Returns a fixture that the rows do not already contain.

    The predictor refuses a fixture it can already see a result for, so a test
    that wants a prediction has to ask for one the data has not decided.

    Parameters:
        rows (list): The matches already on disk.
        extra (tuple): Further names to consider as well.

    Returns:
        tuple: A (home, away) pairing with no result in rows.
    """
    played = {(row["HomeTeam"], row["AwayTeam"]) for row in rows}

    for home in [*TEAMS, *extra]:
        for away in [*TEAMS, *extra]:
            if home != away and (home, away) not in played:
                return home, away

    raise AssertionError("every pairing in the data has been played")


def history_seasons(rows, seasons=2):
    """
    Returns the same fixtures in the seasons before the given one.

    Three seasons are needed before a model can be built: the oldest to fit on,
    the next to validate on, and the newest to predict. A single season leaves
    the fit with nothing to learn from, and two leave it with a validation
    season but no training data. This supplies the history so a builder can hand
    over one season and still get a model that is scoped the way the project
    scopes it.

    The pairings are reused rather than extended, because the clubs playing
    each other a year apart is what a real league does, and it keeps the
    unplayed pairings available for the tests that ask for a fixture to predict.

    Parameters:
        rows (list): The matches of the season under prediction.
        seasons (int): How many seasons to supply ahead of it.

    Returns:
        dict: File name to matches, oldest season first.
    """
    written = {}

    for step in range(seasons, 0, -1):
        earlier = []

        for row in rows:
            copy = dict(row)

            date = pd.to_datetime(copy["Date"], format="%d/%m/%Y")

            copy["Date"] = (
                date - pd.DateOffset(years=step)
            ).strftime("%d/%m/%Y")

            earlier.append(copy)

        written[f"{step}"] = earlier

    return written


def prior_season_name(name, steps=1):
    """
    Returns the file name for a season before the given one.

    Parameters:
        name (str): A season file name, e.g. "2024-2025.csv".
        steps (int): How many seasons back to go.

    Returns:
        str: The earlier season's file name, e.g. "2023-2024.csv".
    """
    stem, _, suffix = name.partition(".")

    start = int(stem.split("-")[0])

    return f"{start - steps}-{start - steps + 1}.{suffix}"


def write_season(directory, rows, name="2024-2025.csv", history=2):
    """
    Writes a season CSV in the DD/MM/YYYY spelling football-data.co.uk uses.

    Earlier seasons are written ahead of it by default, because the model is
    fitted on the seasons before the validation season, and the validation
    season is the one before the season being predicted. Without them there
    would be nothing to fit on.

    Parameters:
        directory (Path): The league directory to write into.
        rows (list): The matches to write.
        name (str): The file name, which is what the season is read from.
        history (int): How many earlier seasons to write, 0 for none.

    Returns:
        Path: The file written for the rows passed in.
    """
    path = directory / name

    pd.DataFrame(rows).to_csv(path, index=False)

    for step, earlier in history_seasons(rows, seasons=history).items():
        pd.DataFrame(earlier).to_csv(
            directory / prior_season_name(name, steps=int(step)),
            index=False,
        )

    return path


def trained_predictor(directory, rows, league="PremierLeague"):
    """
    Writes the rows and returns a predictor for them.

    Parameters:
        directory (Path): The league directory to write into.
        rows (list): The matches to write.
        league (str): The league to build a predictor for.

    Returns:
        Predictor: A predictor over the rows just written.
    """
    from src.predict import Predictor

    write_season(directory, rows)

    return Predictor(league)
