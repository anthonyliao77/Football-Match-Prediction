"""
Writes Understat expected goals into the local football-data.co.uk CSVs.

football-data.co.uk publishes shots and shots on target for every season but no
expected goals, so xG has to come from a second source. This script copies
Understat's xG into the CSVs so the training pipeline can read xG as a plain
column instead of merging an external feed on every run.

Only empty xG cells are written. A value already in the file is left alone, so
running this twice changes nothing the second time, and an xG value that is
already in the file is never replaced.
"""

import argparse
import glob

import pandas as pd

from config import LEAGUES
from src.data_loader import get_season_range, write_csv_atomic
from src.understat_loader import UnderstatDataError, load_understat_data
from src.xg import SOURCE_COLUMN, UNDERSTAT_SOURCE, XG_COLUMNS

# The columns used to line an Understat fixture up with a CSV row. The date is
# matched through a normalised private key rather than the Date column itself,
# because the CSVs store dates as DD/MM/YYYY strings and rewriting them to ISO
# would silently change the on-disk format of a column this script does not own.
DATE_KEY = "_date_key"

MATCH_COLUMNS = [DATE_KEY, "HomeTeam", "AwayTeam"]


def _parse_dates(values):
    """
    Parses dates that may be DD/MM/YYYY or ISO into normalised timestamps.

    ``format="mixed"`` is what allows both spellings in one pass, so the script
    works on a fresh download and on a file it has already written.
    """
    return pd.to_datetime(values, format="mixed", dayfirst=True).dt.normalize()


def _parse_understat_dates(values):
    """
    Parses Understat's dates, which are ISO, not day-first.

    The season CSVs are day-first and need a parser that says so. Handing those
    same ISO dates to that parser reads 2023-08-12 as 12 December, which shifts
    every Understat key by months and leaves the backfill matching nothing
    without raising, so the two conventions are parsed separately.

    A column of real timestamps passes straight through, which is what
    ``load_understat_data`` hands over.
    """
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values).dt.normalize()

    parsed = pd.to_datetime(values, format="%Y-%m-%d", errors="coerce")

    # Some fixtures and hand-built frames spell dates the European way; only fall
    # back when the ISO pass could not read the whole column.
    if parsed.isna().any():
        return pd.to_datetime(values, format="mixed", dayfirst=True).dt.normalize()

    return parsed.dt.normalize()


def _read_csv(path) -> pd.DataFrame:
    """
    Reads one season CSV, adding a normalised date key for matching.

    Every existing cell is kept as the text it was in the file. The season CSVs
    carry over a hundred columns of betting odds, and an odds column with a
    blank in it parses as a float, so reading them normally and writing them
    back turns every 7 into 7.0 and rewrites the whole file. The backfill is
    meant to add three columns, so it reads text and lets the numbers be added
    where they are actually needed.

    The Date column itself is left exactly as the file had it.
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=False).copy()

    frame[DATE_KEY] = _parse_dates(frame["Date"])

    for column in ("HomeTeam", "AwayTeam"):
        frame[column] = frame[column].astype(str)

    # A season that already carries xG was written by an earlier run, so those
    # values are text too and have to become numbers before anything can
    # compare them.
    for column in XG_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # An empty cell is missing data, not a value. Left as an empty string it
    # would read as a source that is already set, and the backfill would fill
    # the xG without ever labelling where it came from.
    if SOURCE_COLUMN in frame.columns:
        frame[SOURCE_COLUMN] = frame[SOURCE_COLUMN].mask(
            frame[SOURCE_COLUMN] == "", pd.NA
        )

    return frame


def _understat_lookup(xg_data: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the match-keyed lookup of Understat xG values.
    """
    lookup = xg_data.rename(
        columns={
            "home_team": "HomeTeam",
            "away_team": "AwayTeam",
        }
    )[["date", "HomeTeam", "AwayTeam", *XG_COLUMNS]].copy()

    for column in ("HomeTeam", "AwayTeam"):
        lookup[column] = lookup[column].astype(str)

    lookup[DATE_KEY] = _parse_understat_dates(lookup.pop("date"))

    for column in XG_COLUMNS:
        lookup[column] = pd.to_numeric(lookup[column], errors="coerce")

    return lookup.drop_duplicates(subset=MATCH_COLUMNS)


def _with_xg_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the frame with the xG and source columns present.

    They are added in a single concat rather than one assignment at a time,
    because the season CSVs carry over a hundred columns and repeated inserts
    leave the frame fragmented enough to slow every later operation down.
    """
    additions = {
        column: (
            pd.Series(float("nan"), index=frame.index)
            if column in XG_COLUMNS
            else pd.Series([None] * len(frame), index=frame.index, dtype=object)
        )
        for column in (*XG_COLUMNS, SOURCE_COLUMN)
        if column not in frame.columns
    }

    if not additions:
        return frame

    return pd.concat([frame, pd.DataFrame(additions)], axis=1)


def _rows_with_xg(frame: pd.DataFrame) -> pd.Series:
    """
    Marks the rows that have at least one of the two xG values.

    A row counts as having xG when either side is present, because Understat
    occasionally supplies one side and not the other.
    """
    if not set(XG_COLUMNS).issubset(frame.columns):
        return pd.Series(False, index=frame.index)

    return frame[list(XG_COLUMNS)].notna().any(axis=1)


def _merge_xg(frame: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Fills the frame's empty xG cells from the lookup, leaving filled cells be.

    A row with no Understat match keeps its nulls, so an unmatched fixture is
    reported rather than silently given a zero.
    """
    prepared = _with_xg_columns(frame)

    for column in XG_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    matched = prepared.merge(
        lookup,
        on=MATCH_COLUMNS,
        how="left",
        suffixes=("", "_understat"),
    )

    resolved = {}
    # Forced to object because a column that is empty in the file comes back from
    # pandas as float64 NaN, and writing a string into a float column raises
    # rather than filling the gap.
    sources = matched[SOURCE_COLUMN].astype(object)

    for column in XG_COLUMNS:
        supplied = matched[f"{column}_understat"]

        # Only the gap is filled, which is what makes this a backfill rather
        # than an overwrite of whatever the file already held.
        filled_from_understat = matched[column].isna() & supplied.notna()

        resolved[column] = matched[column].fillna(supplied).astype(float)

        # Labelled only where Understat actually supplied a value, and only on a
        # row that carries no source yet. A row that already names a source was
        # partly hand-entered, and relabelling the whole row would credit
        # Understat with a value it never provided. xg_source is a row-level
        # label, so a row carrying values from two places can only name the one
        # it already claimed; the filled cell is still visible as a value that
        # was absent before the backfill ran.
        sources.loc[filled_from_understat & sources.isna()] = UNDERSTAT_SOURCE

    resolved[SOURCE_COLUMN] = sources

    # The resolved values replace the originals rather than sitting beside
    # them, so the originals are dropped before the two frames are joined.
    # Leaving them in place would emit duplicate home_xg, home_xg.1,
    # home_xg.2 ... columns.
    matched = matched.drop(
        columns=[
            *[f"{column}_understat" for column in XG_COLUMNS],
            *XG_COLUMNS,
            SOURCE_COLUMN,
        ]
    )

    merged = pd.concat(
        [matched, pd.DataFrame(resolved, index=matched.index)], axis=1
    )

    # The match key is scaffolding and has no place in the file.
    return merged.drop(columns=[DATE_KEY])


def backfill_league_xg(league: str, dry_run: bool = False) -> dict:
    """
    Writes Understat xG into every season CSV of one league.

    All Understat data is loaded and every file is prepared before anything is
    written, so a failure part-way through cannot leave the league with some
    seasons backfilled and others not.

    Parameters:
        league (str): The league directory name (e.g. "PremierLeague").
        dry_run (bool): If True, report the changes without writing the files.

    Returns:
        dict: Counts of the xG cells written, the cells already present, and
        the rows with no Understat match.

    Raises:
        UnderstatDataError: If Understat has no data for the league at all, in
            which case no file is modified.
    """
    directory = LEAGUES[league]["football_data"]

    paths = sorted(glob.glob(f"{directory}/*.csv"))

    if not paths:
        raise FileNotFoundError(
            f"No CSV files found in {directory}. Run update_data.py first."
        )

    frames = [(path, _read_csv(path)) for path in paths]

    combined = pd.concat(
        [frame for _, frame in frames], ignore_index=True
    )

    # The range comes from the same July boundary the rest of the project uses.
    #
    # It used to be worked out here as year-of-date plus one, which is wrong for
    # every match from January to June: a fixture on 30 May 2027 belongs to
    # 2026/2027, not to a 2027/2028 season that does not exist. That only
    # became visible once the season CSVs started holding the rest of the
    # schedule, because until then the newest date in the data was in May and
    # the wrong answer happened to name a season Understat does have.
    #
    # The parsed dates are already on the frame under DATE_KEY, and
    # get_season_range reads the Date column, so it is handed a one-column
    # frame rather than a copy of every column of every season.
    start_year, end_year = get_season_range(
        pd.DataFrame({"Date": combined[DATE_KEY]})
    )

    # Loaded before any write, so a missing Understat season aborts the whole
    # backfill instead of half-populating the league.
    xg_data = load_understat_data(
        league=LEAGUES[league]["understat"],
        start_year=start_year,
        end_year=end_year,
    )

    lookup = _understat_lookup(xg_data)

    summary = {
        "files": 0,
        "rows": 0,
        "xg_filled": 0,
        "xg_already_present": 0,
        "rows_unmatched": 0,
    }

    prepared = []

    for path, frame in frames:
        had_xg = _rows_with_xg(frame)

        merged = _merge_xg(frame, lookup)

        has_xg = _rows_with_xg(merged)

        filled = int((has_xg & ~had_xg).sum())
        unmatched = int((~has_xg).sum())

        summary["files"] += 1
        summary["rows"] += len(merged)
        summary["xg_filled"] += filled
        summary["xg_already_present"] += int(had_xg.sum())
        summary["rows_unmatched"] += unmatched

        prepared.append((path, merged, filled, int(had_xg.sum()), unmatched))

    for path, merged, filled, had_xg, unmatched in prepared:
        verb = "Would backfill" if dry_run else "Backfilled"

        print(
            f"{verb} {path}: {filled} rows gained xG, "
            f"{had_xg} already had xG, "
            f"{unmatched} had no Understat match"
        )

        if dry_run:
            continue

        # Restore the original column order, with the xG columns appended.
        original = pd.read_csv(path, nrows=0).columns.tolist()
        ordered = [column for column in original if column in merged.columns]
        ordered += [
            column for column in merged.columns if column not in ordered
        ]

        write_csv_atomic(path, merged[ordered])

    return summary


def parse_arguments():
    """
    Parse command-line arguments for the xG backfill script.
    """
    parser = argparse.ArgumentParser(
        description="Write Understat expected goals into the local CSVs."
    )

    parser.add_argument(
        "--league",
        choices=["all", *sorted(LEAGUES)],
        default="all",
        help="League to backfill. Defaults to every league.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing the CSVs.",
    )

    return parser.parse_args()


def main():
    """
    Runs the requested xG backfill.
    """
    args = parse_arguments()

    leagues = sorted(LEAGUES) if args.league == "all" else [args.league]

    totals = {
        "files": 0,
        "rows": 0,
        "xg_filled": 0,
        "xg_already_present": 0,
        "rows_unmatched": 0,
    }

    for league in leagues:
        print(f"\n{league}")

        try:
            summary = backfill_league_xg(league, dry_run=args.dry_run)
        except UnderstatDataError as error:
            print(f"  skipped: {error}")
            continue

        for key in totals:
            totals[key] += summary[key]

    verb = "Would write" if args.dry_run else "Wrote"

    print(
        f"\n{verb} {totals['xg_filled']} rows of xG across "
        f"{totals['files']} files "
        f"({totals['xg_already_present']} rows already had xG, "
        f"{totals['rows_unmatched']} rows had no Understat match)."
    )


if __name__ == "__main__":
    main()
