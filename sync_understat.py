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


def _blank_result(frame: pd.DataFrame) -> pd.Series:
    """
    Marks the rows that have no result recorded.

    A row with no result is a fixture that has not been played, and that is the
    only thing this returns True for. An empty string counts as no result, so a
    file written by hand with blanks behaves the same as one written by this
    script.

    Parameters:
        frame (pd.DataFrame): The season as read from the CSV.

    Returns:
        pd.Series: True where the row carries no result.
    """
    if "FTR" not in frame.columns:
        return pd.Series(True, index=frame.index)

    return frame["FTR"].isna() | (frame["FTR"].astype(str).str.strip() == "")


def _reschedule(
    frame: pd.DataFrame, unmatched: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Moves rows on file to the date Understat now gives them.

    A postponed fixture arrives here twice: once as the row on file, still on
    the old date, and once in the fixture list, on the new one. Rewriting the
    row's date settles both, so the season does not end up with two rows for one
    match and a first row that is never played.

    Only rows with no result are eligible, and that condition is the whole of
    the safety argument. A league does play the same pairing more than once in a
    season, but by the time the return leg is listed the first leg has a result,
    so it is not a candidate and the return leg is left to be added as a new
    row. A row that has been played is never re-dated either, because its date
    is the date the match was actually played on, not a fixture that moved.

    Where a pairing has several blank rows and several fixtures, they are paired
    in date order. Two postponed meetings of the same pair then resolve to the
    earlier fixture and the later one, which is the only reading that leaves
    both matches in the file.

    Parameters:
        frame (pd.DataFrame): The season, with a normalised _date column.
        unmatched (pd.DataFrame): Understat fixtures with no row on file.

    Returns:
        tuple: The frame with any moved dates applied, and a list of the moves
        made, each holding the pair, the old date and the new one.
    """
    moves: list[dict] = []

    if not len(unmatched):
        return frame, moves

    blank = frame[_blank_result(frame)]

    candidates: dict[tuple, list] = {}

    for index in blank.index:
        pair = (
            str(blank.at[index, "HomeTeam"]),
            str(blank.at[index, "AwayTeam"]),
        )

        candidates.setdefault(pair, []).append(index)

    for pair, rows in candidates.items():
        rows.sort(key=lambda index: blank.at[index, "_date"])

    for _, fixture in unmatched.sort_values("date").iterrows():
        pair = (fixture["home_team"], fixture["away_team"])

        available = candidates.get(pair, [])

        if not available:
            continue

        index = available.pop(0)

        was = frame.at[index, "_date"]

        if was == fixture["date"]:
            continue

        frame.at[index, "_date"] = fixture["date"]
        frame.at[index, "Date"] = fixture["date"].strftime("%d/%m/%Y")

        moves.append(
            {"pair": f"{pair[0]} v {pair[1]}", "was": was, "now": fixture["date"]}
        )

    return frame, moves


def _played_without_a_result(
    frame: pd.DataFrame, present: pd.DataFrame, missing: pd.DataFrame
) -> pd.DataFrame:
    """
    Lists the played matches that have no result in the CSV.

    Both ways of being missing a result are collected. A fixture with no row on
    file at all, and a fixture whose row is on file and blank because it was
    added before kickoff, are the same problem from the reader's side: the
    season file does not know how that match ended.

    Parameters:
        frame (pd.DataFrame): The season, with a normalised _date column.
        present (pd.DataFrame): Understat fixtures already matched to a row.
        missing (pd.DataFrame): Understat fixtures with no row on file.

    Returns:
        pd.DataFrame: The played fixtures that still have no result on file.
    """
    blank = _blank_result(frame)

    keys = {
        (row._date, str(row.HomeTeam), str(row.AwayTeam)): blank.at[key]
        for key, row in frame.iterrows()
    }

    def lacks_a_result(row) -> bool:
        key = (row["date"], row["home_team"], row["away_team"])

        if key not in keys:
            return True

        return bool(keys[key])

    played = pd.concat(
        [present[present["played"]], missing[missing["played"]]],
        ignore_index=True,
    )

    return played[played.apply(lacks_a_result, axis=1)]


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


def sync_league(league: str, season: str, dry_run: bool = False) -> dict:
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

    # Presence is matched on the fixture itself: date, home and away. A fixture
    # that has moved to a new date is a different key, so it is dealt with by
    # the reschedule pass below rather than being appended a second time.
    #
    # The date column is joined rather than assigned. A football-data.co.uk
    # season file is 117 columns wide, and pandas reads that as a frame of many
    # separate blocks, at which point writing a column into it is a per-block
    # copy and pandas says so. Joining builds a new frame instead.
    frame = pd.concat(
        [frame, _date_key(frame["Date"]).rename("_date")], axis=1
    )

    existing = set(
        zip(
            frame["_date"],
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

    unmatched = known_fixtures[~known_fixtures.apply(is_present, axis=1)]

    # A postponed fixture. Understat publishes the new date and the CSV still
    # holds the old one, so the pair is on file at a date Understat no longer
    # lists. Rewriting that row's date is what keeps the season from acquiring a
    # second row for one match and a first row that is never played.
    #
    # Only rows with no result are eligible. That single condition is what makes
    # this safe: a league does play the same pairing more than once in a season,
    # but by the time the return leg comes round the first one has a result, so
    # it is not a candidate and the return leg is added as a new row. A row that
    # has been played is never rewritten either, because its date is the date it
    # was actually played on.
    frame, rescheduled = _reschedule(frame, unmatched)

    existing = set(
        zip(
            frame["_date"],
            frame["HomeTeam"].astype(str),
            frame["AwayTeam"].astype(str),
        )
    )

    present = known_fixtures[known_fixtures.apply(is_present, axis=1)]
    missing = known_fixtures[~known_fixtures.apply(is_present, axis=1)]

    added = missing[~missing["played"]]

    # A played match is missing its result in two ways: the file has no row for
    # it at all, or the row is there and still blank because the fixture was
    # added before kickoff and the results have not been downloaded since. The
    # second is the common one once a season is running, and it is the one that
    # matters, because a blank row reads as a match that has not been played: it
    # is dropped from the features and from the Elo, and the club's recent form
    # quietly goes stale.
    #
    # Both are reported together as one instruction. Results come from
    # football-data.co.uk and are not written here, so the count is how far
    # behind that download is.
    settled = _played_without_a_result(frame, present, missing)

    print(f"{league} {season}: {len(known_fixtures)} fixtures listed by Understat")
    print(f"  {len(present)} already in {path.rsplit('/', 1)[-1]}")
    print(f"  {len(added)} pending fixtures to add")
    print(f"  {len(rescheduled)} rescheduled fixtures re-dated")
    print(f"  {len(settled)} played matches with a blank result in the CSV")

    if len(rescheduled):
        print("\n  Fixtures that moved date. The date on file was rewritten:")
        for row in rescheduled:
            print(f"    {row['was'].date()} -> {row['now'].date()}  {row['pair']}")

    if len(settled):
        print("\n  Played matches whose result is not on file. Results come from")
        print("  football-data.co.uk, so these are reported rather than written.")
        print("  Refresh the season download, then rerun:")
        for _, row in settled.iterrows():
            print(f"    {row['date'].date()} {row['home_team']} v {row['away_team']}")

    if dry_run or (not len(added) and not len(rescheduled)):
        return {
            "added": len(added),
            "already_present": len(present),
            "rescheduled": len(rescheduled),
            "needs_results": len(settled),
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

    # A re-dated fixture is a change on its own, so a run can reach here with
    # nothing to add, and concat is skipped rather than fed an empty frame.
    #
    # The normalised _date column is dropped before the concat as well as after
    # it. The added rows have no counterpart to it, so leaving it in would have
    # pandas fill the whole column with NaT and widen the datetime dtype, and it
    # is working state that must not reach the file. The Date column itself was
    # already corrected in place by the reschedule pass.
    if fixtures_to_write:
        combined = pd.concat(
            [frame.drop(columns=["_date"]), pd.DataFrame(fixtures_to_write)],
            ignore_index=True,
        )
    else:
        combined = frame.drop(columns=["_date"])

    # The fixture columns keep their place in the file, and anything new the
    # added rows introduced is appended, so the shape stays uniform.
    original = pd.read_csv(path, nrows=0).columns.tolist()
    ordered = [column for column in original if column in combined.columns]
    ordered += [
        column for column in combined.columns if column not in ordered
    ]

    write_csv_atomic(path, combined[ordered])

    written = f"{len(fixtures_to_write)} pending fixtures and "
    written += f"{len(rescheduled)} re-dated fixtures" if rescheduled else ""

    print(f"\n  Wrote {written} to {path}")

    return {
        "added": len(fixtures_to_write),
        "already_present": len(present),
        "rescheduled": len(rescheduled),
        "needs_results": len(settled),
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
            sync_league(league, season, dry_run=arguments.dry_run)
        except (UnderstatUnavailable, FileNotFoundError) as error:
            print(f"{league}: {error}")


if __name__ == "__main__":
    main()
