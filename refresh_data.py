"""
Refreshes the season CSVs from Understat in one command.

A season file needs three things kept current, and they come from two places.
The fixtures still to come and the xG of the matches already played both come
from Understat, and add_fixtures.py and backfill_xg.py each know how to get one
of them. The results do not: those come from a football-data.co.uk download,
which is a manual step this script cannot do and does not pretend to do.

So this is an orchestrator, not new logic. It calls the two existing entry
points, aggregates what they report, and ends by saying the one thing the
person running it needs to know, which is how far behind their results download
is. That number is the reason to run the command rather than either script
alone: neither of them can see that a fixture has been played since the file was
last downloaded.

Order is fixtures then xG. The two do not overlap, since the fixtures step only
adds rows with no result and the xG step only fills empty xG cells on rows that
have one, so either order would give the same files. It is written this way
because a fixture moved by the first step is picked up as a cell to fill by the
second, and that reads in the order it happens.

A league that fails does not stop the others. Understat is an undocumented
endpoint with no published availability, so one league being unreadable is a
normal enough event that it should not cost you the other two. The exit status
is non-zero if any league failed, which is what makes this safe to put on a
schedule.
"""

import argparse
import sys

from add_fixtures import _default_season, add_fixtures
from backfill_xg import backfill_league_xg
from config import LEAGUES
from src.understat_client import UnderstatUnavailable
from src.understat_loader import UnderstatDataError

# The xG summary keys worth reporting, and what to call them in the table.
XG_FIELDS = [
    ("xg_filled", "xG written"),
    ("rows", "rows read"),
    ("rows_unmatched", "no Understat match"),
]


def parse_arguments() -> argparse.Namespace:
    """
    Reads the command line.

    Returns:
        argparse.Namespace: The requested league, season, and whether to write.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Refresh the fixtures and the xG of every league from Understat. "
            "Results still come from a football-data.co.uk download."
        )
    )

    parser.add_argument(
        "--league",
        choices=["all", *LEAGUES],
        default="all",
        help="League to refresh. Defaults to every league.",
    )

    parser.add_argument(
        "--season",
        default=None,
        help=(
            "Season to fill fixtures for, e.g. 2026/2027. Defaults to the most "
            "recent season file in the league. The xG step always covers the "
            "whole league, since it only fills cells that are empty."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing the CSVs.",
    )

    return parser.parse_args()


def refresh_league(league: str, season: str | None, dry_run: bool) -> dict:
    """
    Refreshes one league's fixtures and then its xG.

    Parameters:
        league (str): The league directory name, e.g. "PremierLeague".
        season (str | None): Season to fill, or None to take the newest.
        dry_run (bool): If True, report without writing.

    Returns:
        dict: The fixture and xG counts, and any error raised for this league.
    """
    directory = LEAGUES[league]["football_data"]

    if season is None:
        season = _default_season(directory)

    row = {
        "league": league,
        "season": season,
        "added": 0,
        "rescheduled": 0,
        "needs_results": 0,
        "xg_filled": 0,
        "rows": 0,
        "rows_unmatched": 0,
        "error": None,
    }

    if season is None:
        row["error"] = f"no season CSVs found in {directory}"

        return row

    try:
        fixtures = add_fixtures(league, season, dry_run=dry_run)

        row["added"] = fixtures["added"]
        row["rescheduled"] = fixtures["rescheduled"]
        row["needs_results"] = fixtures["needs_results"]
    except (UnderstatUnavailable, FileNotFoundError) as error:
        row["error"] = str(error)

        return row

    try:
        xg = backfill_league_xg(league, dry_run=dry_run)

        for key in ("xg_filled", "rows", "rows_unmatched"):
            row[key] = xg[key]
    except UnderstatDataError as error:
        row["error"] = f"xG: {error}"

    return row


def report(rows: list[dict], dry_run: bool) -> int:
    """
    Prints the summary and returns the number of leagues that failed.

    Parameters:
        rows (list[dict]): One row per league, from refresh_league.
        dry_run (bool): Whether this was a dry run.

    Returns:
        int: How many leagues failed.
    """
    verb = "Would change" if dry_run else "Changed"

    print(f"\n{verb}, by league:")
    print("A stale count is a played match with no result on file. An xG count")
    print("of zero on a fresh download is the normal case: it only fills cells")
    print("that are empty.\n")

    header = f"{'League':<16}{'Fixtures':>10}{'Re-dated':>10}{'xG':>8}{'Stale':>8}"

    print(header)
    print("-" * len(header))

    failed = 0
    stale = 0

    for row in rows:
        if row["error"]:
            failed += 1
            print(f"{row['league']:<16}  FAILED: {row['error']}")
            continue

        stale += row["needs_results"]

        print(
            f"{row['league']:<16}"
            f"{row['added']:>10}"
            f"{row['rescheduled']:>10}"
            f"{row['xg_filled']:>8}"
            f"{row['needs_results']:>8}"
        )

    if stale:
        print(
            f"\n{stale} played "
            f"{'match has' if stale == 1 else 'matches have'} a result upstream "
            f"but a blank result in the CSV."
        )
        print(
            "Download the season from football-data.co.uk and replace the file, "
            "then run this again."
        )
    else:
        print("\nEvery played match on file has its result.")

    if failed:
        print(
            f"\n{failed} of {len(rows)} leagues could not be refreshed. "
            "Understat publishes no API and no availability, so this is usually "
            "temporary; rerun before trusting the data."
        )

    return failed


def main() -> int:
    """
    Runs the refresh.

    Returns:
        int: 0 if every league was refreshed, 1 otherwise.
    """
    arguments = parse_arguments()

    leagues = list(LEAGUES) if arguments.league == "all" else [arguments.league]

    rows = [
        refresh_league(league, arguments.season, arguments.dry_run)
        for league in leagues
    ]

    failed = report(rows, arguments.dry_run)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
