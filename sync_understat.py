"""
Brings the local season CSVs up to date with Understat in one pass.

Two kinds of thing are missing from a season file, and they come from the same
place now that they are filled from the same source.

The season's remaining fixtures. football-data.co.uk publishes a season as it is
played, so a CSV downloaded partway through holds only the matches that have
been decided. Understat lists the whole season at once, including the fixtures
still to come, each with its date and sides but no score.

The results of matches that have since been played. A fixture added before
kickoff leaves a row with a blank result, and that row is not a neutral blank:
it reads as a match that has not happened, so it is dropped from the features
and from the Elo and the club's recent form goes stale with nothing to say so.
Understat publishes a score as soon as the match is over, so the result is
filled in here rather than waiting for the next manual download.

The cost of the two is not the same, and that asymmetry is why they are handled
differently. A score is in the league payload this script has already read, so
filling one costs nothing. Shots are not: Understat's league payload has no
shot counts on it at all, and they cost one request per match. So they are only
fetched for a match that is actually being filled, and never to re-derive what
the file already holds.

A match is written whole or not at all, which is the rule everything else bends
around. src/features.py turns a missing number into a zero on purpose, so a row
given a score and no shot counts is not a row with two blanks in it but a row
claiming the club managed no shots in that match. Nothing would fail. The
features would sum, the model would predict, and the answer would be quietly
wrong by about twenty shots.

Existing rows are never overwritten. Any row that already has a result keeps it,
whatever the other source says, so the football-data.co.uk download stays the
authority on every row that has one and a disagreement between the two is
counted and shown rather than silently resolved. Running this twice changes
nothing the second time, and spends no requests.
"""

import argparse
import glob
import os

import pandas as pd

from config import LEAGUES, TEAM_NAME_MAP
from src.data_loader import prediction_season, write_csv_atomic
from src.results import (
    FILLED_COLUMNS,
    RESULT_SOURCE_COLUMN,
    SHOT_COLUMNS,
    UNDERSTAT_RESULT_SOURCE,
)
from src.understat_client import (
    UnderstatUnavailable,
    get_league_fixtures,
    get_match_shots,
)

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


def _dates_of(rows: list[dict]) -> list:
    """
    Plucks the dates out of the matches deliberately left alone.

    Reported alongside the count because the count alone cannot answer the
    question the refresh actually needs to ask. A match left unwritten an hour
    ago is one someone will write tonight; one left unwritable for three weeks
    is a fault nobody has looked at, and only the date tells those apart.

    Parameters:
        rows (list[dict]): The unwritten matches, each carrying a date.

    Returns:
        list: Their dates, in the order they were found.
    """
    return [row["date"] for row in rows]


def _shot_gaps(frame: pd.DataFrame, index) -> list[str]:
    """
    Names the shot columns a row that already has a result is still missing.

    Parameters:
        frame (pd.DataFrame): The season as read from the CSV.
        index: The row to look at.

    Returns:
        list[str]: The empty shot columns, or an empty list where the row is
        whole.
    """
    present = [column for column in FILLED_COLUMNS if column in frame.columns]
    absent = [column for column in present if pd.isna(frame.at[index, column])]

    return [column for column in absent if column in SHOT_COLUMNS]


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


def _local_index(frame: pd.DataFrame) -> dict:
    """
    Indexes the rows on file by the key a fixture is matched on.

    Parameters:
        frame (pd.DataFrame): The season, with a normalised _date column.

    Returns:
        dict: (date, home team, away team) to the row's index.
    """
    return {
        (row._date, str(row.HomeTeam), str(row.AwayTeam)): index
        for index, row in frame.iterrows()
    }


def _looks_unplayed(home_goals, away_goals, shots: dict) -> bool:
    """
    Reports whether a 0-0 is a match nobody took part in.

    A goalless draw with not one shot between the two teams is not a football
    match, it is a fixture Understat has listed and scored 0-0 because the game
    was abandoned, postponed to a date it has not published, or never played. The
    leagues do record those, and writing one of them as a played 0-0 would add a
    match to the season that never happened, complete with a form entry and an
    Elo result for both clubs.

    The test is deliberately narrow. A real goalless draw has shots, so only the
    no-shots-at-all case is caught, and a match abandoned after kickoff still
    counts as played, which is the right way round: guessing at the rest would
    mean skipping real results.

    Parameters:
        home_goals: Goals scored by the home team.
        away_goals: Goals scored by the away team.
        shots (dict): The counts from get_match_shots.

    Returns:
        bool: True where this should be left unwritten.
    """
    if home_goals or away_goals:
        return False

    return not (shots["home_shots"] or shots["away_shots"])


def _plan_results(
    frame: pd.DataFrame, played: pd.DataFrame
) -> tuple[list[dict], list[dict], list[dict], list[dict], int]:
    """
    Decides which played matches to write a result for, and fetches the shots.

    Three things are deliberately different here. A match already on file with a
    result is never touched, so the football-data.co.uk download stays the
    authority on every row that has one. A match with no row at all, or a row
    still blank, is filled from Understat. And a match whose two sources
    disagree is counted, not resolved, because a rewrite would mean trusting one
    number over another on no evidence and no way back.

    Every match is written whole or not at all. That is the constraint the whole
    design turns on. src/features.py coerces a missing number to zero on purpose,
    so a row given a score and no shot counts is not a row with two blanks in it
    but a row that tells the model the club managed no shots and no shots on
    target in that match. The error would not surface anywhere: the features
    would sum, the model would predict, and the result would be quietly wrong by
    about twenty shots. So if the shot counts cannot be had, neither is the score.

    One request is spent per match filled, and none per match already on file.

    Parameters:
        frame (pd.DataFrame): The season, with a normalised _date column.
        played (pd.DataFrame): Understat's fixtures that have been played.

    Returns:
        tuple: The matches to write, the ones deliberately left alone, the ones
        whose scores the two sources disagree about, the rows that already carry
        a result but are missing shot counts, and how many match requests were
        spent.
    """
    local = _local_index(frame)
    blank = _blank_result(frame)

    to_write: list[dict] = []
    abandoned: list[dict] = []
    disagreements: list[dict] = []
    partial: list[dict] = []
    requests = 0

    for _, fixture in played.iterrows():
        key = (
            fixture["date"],
            str(fixture["home_team"]),
            str(fixture["away_team"]),
        )

        home_goals = fixture["home_goals"]
        away_goals = fixture["away_goals"]

        outcome = _outcome(home_goals, away_goals)

        if outcome is None:
            continue

        index = local.get(key)

        if index is not None and not blank.at[index]:
            theirs = outcome
            ours = str(frame.at[index, "FTR"]).strip()

            if ours and ours != theirs:
                disagreements.append({
                    "pair": f"{key[1]} v {key[2]}",
                    "date": key[0],
                    "ours": ours,
                    "theirs": theirs,
                })

            gaps = _shot_gaps(frame, index)

            if gaps:
                # A row that already carries a result is left exactly as it is,
                # and that rule is what keeps the two vendors out of the same
                # match. It is also how a half-complete row stays half-complete
                # forever, and the features read a blank shot count as zero, so
                # this is reported rather than quietly tolerated. The fix is a
                # decision about which source that row belongs to, not something
                # to settle here.
                partial.append({
                    "pair": f"{key[1]} v {key[2]}",
                    "date": key[0],
                    "missing": gaps,
                })

            continue

        if pd.isna(fixture["understat_id"]):
            abandoned.append({
                "pair": f"{key[1]} v {key[2]}",
                "date": key[0],
                "reason": "Understat listed it without an id to read shots from",
            })

            continue

        requests += 1

        try:
            shots = get_match_shots(fixture["understat_id"])
        except UnderstatUnavailable as error:
            # The score is in hand and is not written, because a row with a
            # score and no shot counts is worse than a blank row.
            abandoned.append({
                "pair": f"{key[1]} v {key[2]}",
                "date": key[0],
                "reason": str(error),
            })

            continue

        if _looks_unplayed(home_goals, away_goals, shots):
            abandoned.append({
                "pair": f"{key[1]} v {key[2]}",
                "date": key[0],
                "reason": "0-0 with no shots from either side, so probably not "
                "a match that was played",
            })

            continue

        to_write.append({
            "key": key,
            "index": index,
            "date": key[0],
            "home_team": key[1],
            "away_team": key[2],
            "FTHG": int(home_goals),
            "FTAG": int(away_goals),
            "FTR": outcome,
            "HS": shots["home_shots"],
            "AS": shots["away_shots"],
            "HST": shots["home_on_target"],
            "AST": shots["away_on_target"],
        })

    return to_write, abandoned, disagreements, partial, requests


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

    played_upstream = pd.concat(
        [present[present["played"]], missing[missing["played"]]],
        ignore_index=True,
    )

    # A match Understat has played and the file has no result for is filled in,
    # which is what makes the season self-healing. Both ways of missing one are
    # covered: a fixture the file never heard of, and a row that is on file and
    # still blank because it was added before kickoff. The second is the common
    # case once a season is running, and it is the one that matters, because a
    # blank row reads as a match that has not happened: it is dropped from the
    # features and from the Elo, and the club's recent form goes stale quietly.
    # A match can only be written whole. A season file that does not carry every
    # one of those columns cannot hold a complete match, so nothing is filled
    # rather than a result written without the shot counts beside it. The
    # football-data.co.uk download is what supplies the columns, and every file
    # this project has seen carries all seven.
    absent = [column for column in FILLED_COLUMNS if column not in frame.columns]

    if absent:
        to_write, abandoned, disagreements, partial, requests = [], [], [], [], 0

        print(
            f"  WARNING: {path.rsplit('/', 1)[-1]} has no {', '.join(absent)}, so "
            f"no result can be written without leaving a half-complete match. "
            f"Re-download the season from football-data.co.uk."
        )
    else:
        to_write, abandoned, disagreements, partial, requests = _plan_results(
            frame, played_upstream
        )

    filling = [row for row in to_write if row["index"] is not None]
    appending = [row for row in to_write if row["index"] is None]

    for row in filling:
        for column in FILLED_COLUMNS:
            frame.at[row["index"], column] = row[column]

        if RESULT_SOURCE_COLUMN in frame.columns:
            frame.at[row["index"], RESULT_SOURCE_COLUMN] = (
                UNDERSTAT_RESULT_SOURCE
            )

    print(f"{league} {season}: {len(known_fixtures)} fixtures listed by Understat")
    print(f"  {len(present)} already in {path.rsplit('/', 1)[-1]}")
    print(f"  {len(added)} pending fixtures to add")
    print(f"  {len(rescheduled)} rescheduled fixtures re-dated")
    print(f"  {len(filling)} played matches filled in from Understat")
    print(f"  {len(appending)} played matches added in from Understat")
    print(f"  {requests} Understat match requests spent on shot counts")

    if len(rescheduled):
        print("\n  Fixtures that moved date. The date on file was rewritten:")
        for row in rescheduled:
            print(f"    {row['was'].date()} -> {row['now'].date()}  {row['pair']}")

    if partial:
        print("\n  Rows with a result but missing shot counts. Left exactly as")
        print("  they are, since filling them would put Understat's shots beside")
        print("  another source's goals, but a blank reads as zero downstream:")
        for row in partial:
            print(f"    {row['date'].date()} {row['pair']}  "
                  f"no {', '.join(row['missing'])}")

    if disagreements:
        print("\n  Matches where the two sources disagree. The value on file is")
        print("  kept and nothing is written, but the difference is worth seeing:")
        for row in disagreements:
            print(f"    {row['date'].date()} {row['pair']}  "
                  f"file {row['ours']} vs Understat {row['theirs']}")

    if abandoned:
        print("\n  Played matches left without a result:")
        for row in abandoned:
            print(f"    {row['date'].date()} {row['pair']}")
            print(f"      {row['reason']}")

    writes_pending = bool(
        len(added) or rescheduled or filling or appending or partial
    )

    if dry_run or not writes_pending:
        return {
            "added": len(added),
            "already_present": len(present),
            "rescheduled": len(rescheduled),
            "needs_results": len(abandoned),
            "unwritten_dates": _dates_of(abandoned),
            "results_filled": len(filling),
            "results_added": len(appending),
            "disagreements": len(disagreements),
            "partial_results": len(partial),
            "match_requests": requests,
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

    if RESULT_SOURCE_COLUMN in frame.columns:
        wanted.append(RESULT_SOURCE_COLUMN)

    rows_to_write = []

    for _, row in added.iterrows():
        record = {
            "Date": row["date"].strftime("%d/%m/%Y"),
            "Div": div,
            "HomeTeam": row["home_team"],
            "AwayTeam": row["away_team"],
        }

        for column in wanted:
            record.setdefault(column, None)

        rows_to_write.append(
            {column: record[column] for column in wanted}
        )

    # A played match with no row on file is appended with its result, exactly
    # as a pending fixture is appended without one. Both are a match the season
    # was missing, and the difference is only what the row can say about it.
    for row in appending:
        record = {
            "Date": row["date"].strftime("%d/%m/%Y"),
            "Div": div,
            "HomeTeam": row["home_team"],
            "AwayTeam": row["away_team"],
            RESULT_SOURCE_COLUMN: UNDERSTAT_RESULT_SOURCE,
        }

        for column in FILLED_COLUMNS:
            record[column] = row[column]

        for column in wanted:
            record.setdefault(column, None)

        rows_to_write.append(
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
    if rows_to_write:
        combined = pd.concat(
            [frame.drop(columns=["_date"]), pd.DataFrame(rows_to_write)],
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

    parts = []

    if rows_to_write:
        parts.append(f"{len(rows_to_write)} new rows")

    if rescheduled:
        parts.append(f"{len(rescheduled)} re-dated fixtures")

    if filling:
        parts.append(f"{len(filling)} results filled in")

    print(f"\n  Wrote {', '.join(parts)} to {path}")

    return {
        "added": len(added),
        "already_present": len(present),
        "rescheduled": len(rescheduled),
        "needs_results": len(abandoned),
        "unwritten_dates": _dates_of(abandoned),
        "results_filled": len(filling),
        "results_added": len(appending),
        "disagreements": len(disagreements),
        "partial_results": len(partial),
        "match_requests": requests,
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
