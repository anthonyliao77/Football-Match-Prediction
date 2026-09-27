"""
Writes the remaining fixtures of a season into the local season CSVs.

football-data.co.uk publishes a season as it is played, so a CSV downloaded
partway through a season holds only the matches that have been decided. Understat
lists the whole season at once, including the fixtures still to come, each with
its date and sides but no score. This script copies those pending fixtures in, so
the CSV holds the season's schedule rather than only its results.

A pending fixture is written with no result. That is what keeps it out of the
model: src/training.py and src/predict.py drop rows with no FTR before the
feature and Elo passes, and the season being predicted is held out of the fit
anyway. A fixture is a date and two clubs, not a match, and the pipeline has no
way to read one as a result.

Existing rows are never touched. A played match is never overwritten, and a
fixture already in the file is left exactly as it is, so running this twice
changes nothing the second time.

A played match that Understat has and the CSV does not is reported rather than
written. Results belong to football-data.co.uk, and a fixture that has been
played but is missing from the CSV is a gap in that source, not a scheduling
question, so it is named instead of being filled in from a second source.
"""

import argparse
import glob
import os

import pandas as pd

from config import LEAGUES, TEAM_NAME_MAP
from src.data_loader import prediction_season, write_csv_atomic
from src.understat_client import UnderstatUnavailable, get_league_fixtures

# The xG columns come from src.xg, which is where backfill_xg.py writes them.
from src.xg import SOURCE_COLUMN, XG_COLUMNS

# The columns a pending fixture is built from. Everything else in the file is
# left empty: a fixture that has not been played has no shots, no xG and no
# odds, and inventing a zero for any of them would be a measurement.
FIXTURE_COLUMNS = ["Date", "Div", "HomeTeam", "AwayTeam"]

# The columns written only when the file already has them, so an added fixture
# is the same shape as the rows around it.
OPTIONAL_COLUMNS = ["FTHG", "FTAG", "FTR", "HS", "AS", "HST", "AST"]


def _season_file(directory: str, season: str) -> str | None:
    """
    Returns the CSV holding a season.

    The season file is named for the two years it spans, so it is matched
    exactly. A substring match on the start year alone would also match the
    previous season's file, since 2026 appears in 2025-2026.csv, and the
    fixtures would then be added to the wrong season.

    Parameters:
        directory (str): The league's CSV directory.
        season (str): The season label, e.g. "2026/2027".

    Returns:
        str: The path, or None when the season has no file yet.
    """
    start = season.split("/")[0]

    expected = f"{directory}/{start}-{int(start) + 1}.csv"

    if os.path.exists(expected):
        return expected

    return None


def _date_key(values) -> pd.Series:
    """
    Normalises dates for matching without changing what is stored.

    The CSVs hold DD/MM/YYYY strings and Understat holds timestamps, so the two
    are reduced to a plain date to compare. Rewriting the Date column itself
    would change the on-disk spelling of a column this script does not own.

    Parameters:
        values: The dates to normalise, in any parseable spelling.

    Returns:
        pd.Series: The normalised dates.
    """
    return pd.to_datetime(values, dayfirst=True).dt.normalize()


def _outcome(home_goals, away_goals) -> str | None:
    """
    Returns the result letter for a score.

    Parameters:
        home_goals: Goals scored by the home team.
        away_goals: Goals scored by the away team.

    Returns:
        str: "H", "D" or "A", or None where there is no score.
    """
    if pd.isna(home_goals) or pd.isna(away_goals):
        return None

    if home_goals > away_goals:
        return "H"

    if home_goals == away_goals:
        return "D"

    return "A"


def _div_for(frame: pd.DataFrame, league: str) -> str:
    """
    Returns the division code to write on an added fixture.

    Parameters:
        frame (pd.DataFrame): The season as it currently stands.
        league (str): The league key, for its configured division.

    Returns:
        str: The division code, taken from the file where it has one.
    """
    if "Div" in frame.columns:
        present = frame["Div"].dropna().unique()

        if len(present) == 1:
            return present[0]

    return LEAGUES[league]["div"]


def add_fixtures(league: str, season: str, dry_run: bool = False) -> dict:
    """
    Adds the season's pending fixtures to one league's CSV.

    Everything is prepared before anything is written, so a failure part way
    through cannot leave one league updated and another not.

    Parameters:
        league (str): The league directory name, e.g. "PremierLeague".
        season (str): The season label, e.g. "2026/2027".
        dry_run (bool): If True, report what would change without writing.

    Returns:
        dict: Counts of the fixtures added, already present, and the played
        matches missing from the CSV, plus the names that could not be paired.

    Raises:
        FileNotFoundError: If the season has no CSV to add fixtures to.
    """
    directory = LEAGUES[league]["football_data"]

    path = _season_file(directory, season)

    if path is None:
        raise FileNotFoundError(
            f"No CSV for {season} in {directory}. Download the season from "
            f"football-data.co.uk first, so there is a file to add fixtures to."
        )

    frame = pd.read_csv(path)

    fixtures = get_league_fixtures(LEAGUES[league]["understat"], season)

    for column in ("home_team", "away_team"):
        fixtures[column] = fixtures[column].replace(TEAM_NAME_MAP).astype(str)

    # A team the CSV has never seen is a pairing this script cannot make sense
    # of, so it is named rather than written as a club of its own. Two spellings
    # of one club would split that club's Elo rating and rolling form in two.
    known = set(frame["HomeTeam"].dropna().astype(str)) | set(
        frame["AwayTeam"].dropna().astype(str)
    )

    unknown = sorted(
        (set(fixtures["home_team"]) | set(fixtures["away_team"])) - known
    )

    if unknown:
        print(
            f"WARNING: {league} has teams the CSV does not: {unknown}. They "
            f"are left out. Add them to config.TEAM_NAME_MAP or the season CSV."
        )

    known_fixtures = fixtures[
        fixtures["home_team"].isin(known) & fixtures["away_team"].isin(known)
    ]

    # Presence is matched on the fixture itself. A postponed match that was
    # replayed on another date is a different key, so it is reported below
    # rather than being added a second time under a new date.
    existing = set(
        zip(
            _date_key(frame["Date"]),
            frame["HomeTeam"].astype(str),
            frame["AwayTeam"].astype(str),
        )
    )

    def is_present(row) -> bool:
        return (
            row["date"],
            row["home_team"],
            row["away_team"],
        ) in existing

    present = known_fixtures[known_fixtures.apply(is_present, axis=1)]
    missing = known_fixtures[~known_fixtures.apply(is_present, axis=1)]

    added = missing[~missing["played"]]
    undecided = missing[missing["played"]]

    print(f"{league} {season}: {len(known_fixtures)} fixtures listed by Understat")
    print(f"  {len(present)} already in {path.rsplit('/', 1)[-1]}")
    print(f"  {len(added)} pending fixtures to add")
    print(f"  {len(undecided)} played matches missing from the CSV")

    if len(undecided):
        print("\n  Played matches the CSV does not hold. Results come from")
        print("  football-data.co.uk, so these are reported rather than written:")
        for _, row in undecided.iterrows():
            print(f"    {row['date'].date()} {row['home_team']} v {row['away_team']}")

    if dry_run or not len(added):
        return {
            "added": len(added),
            "already_present": len(present),
            "played_missing": len(undecided),
            "unknown_teams": unknown,
        }

    div = _div_for(frame, league)

    # Only the columns the file already carries are written. A season that has
    # never been given an xG column should not acquire an empty one here: the
    # pipeline reads what is there, and inventing a column is a change to the
    # file's shape that nothing asked for.
    wanted = [column for column in FIXTURE_COLUMNS if column in frame.columns]

    if "Div" not in frame.columns:
        wanted.append("Div")

    wanted += [
        column for column in OPTIONAL_COLUMNS if column in frame.columns
    ]

    wanted += [
        column for column in XG_COLUMNS if column in frame.columns
    ]

    if SOURCE_COLUMN in frame.columns:
        wanted.append(SOURCE_COLUMN)

    fixtures_to_write = []

    for _, row in added.iterrows():
        record = {
            "Date": row["date"].strftime("%d/%m/%Y"),
            "Div": div,
            "HomeTeam": row["home_team"],
            "AwayTeam": row["away_team"],
        }

        for column in wanted:
            record.setdefault(column, None)

        fixtures_to_write.append(
            {column: record[column] for column in wanted}
        )

    combined = pd.concat(
        [frame, pd.DataFrame(fixtures_to_write)],
        ignore_index=True,
    )

    # The fixture columns keep their place in the file, and anything new the
    # added rows introduced is appended, so the shape stays uniform.
    original = pd.read_csv(path, nrows=0).columns.tolist()
    ordered = [column for column in original if column in combined.columns]
    ordered += [column for column in combined.columns if column not in ordered]

    write_csv_atomic(path, combined[ordered])

    print(f"\n  Added {len(fixtures_to_write)} pending fixtures to {path}")

    return {
        "added": len(fixtures_to_write),
        "already_present": len(present),
        "played_missing": len(undecided),
        "unknown_teams": unknown,
    }


def _newest_season_file(directory: str) -> str | None:
    """
    Returns the CSV for the most recent season in a directory.

    Parameters:
        directory (str): The league's CSV directory.

    Returns:
        str: The path, or None when the directory holds no CSVs.
    """
    paths = sorted(glob.glob(f"{directory}/*.csv"))

    if not paths:
        return None

    return paths[-1]


def _default_season(directory: str) -> str | None:
    """
    Returns the season label of the most recent season in a directory.

    Parameters:
        directory (str): The league's CSV directory.

    Returns:
        str: The season label, or None when the directory holds no CSVs.
    """
    path = _newest_season_file(directory)

    if path is None:
        return None

    dates = pd.to_datetime(
        pd.read_csv(path, usecols=["Date"])["Date"],
        dayfirst=True,
    )

    return prediction_season(pd.DataFrame({"Date": dates}))


def parse_arguments():
    """Parses the command line."""
    parser = argparse.ArgumentParser(
        description="Add a season's pending fixtures to the local CSVs."
    )
    parser.add_argument(
        "--league",
        choices=["all", *LEAGUES],
        default="all",
        help="League to add fixtures for. Defaults to every league.",
    )
    parser.add_argument(
        "--season",
        default=None,
        help=(
            "Season label, e.g. 2026/2027. Defaults to the newest season in "
            "the CSVs, which is the one being predicted."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing the CSVs.",
    )

    return parser.parse_args()


def main():
    """Runs the script."""
    arguments = parse_arguments()

    leagues = list(LEAGUES) if arguments.league == "all" else [arguments.league]

    for league in leagues:
        if league != leagues[0]:
            print()

        season = arguments.season

        if season is None:
            season = _default_season(LEAGUES[league]["football_data"])

        if season is None:
            print(f"{league}: no season CSVs found.")
            continue

        try:
            add_fixtures(league, season, dry_run=arguments.dry_run)
        except (UnderstatUnavailable, FileNotFoundError) as error:
            print(f"{league}: {error}")


if __name__ == "__main__":
    main()
