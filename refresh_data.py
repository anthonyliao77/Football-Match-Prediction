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

A league that failed is counted as having an unknown number of stale matches,
not as having none, and the table says so with a dash. The alternative was a
refresh that reported every league as clean while one of them had not been
looked at, which is the failure mode a staleness check exists to prevent.

--require-fresh extends the exit status to cover a stale results download. A
failed refresh is loud, but the likelier way for a scheduled run to be quietly
useless is for both halves to succeed while the results download is a fortnight
old, and nothing about that run looks wrong.
"""

import argparse
import sys
from dataclasses import dataclass

from add_fixtures import _default_season, add_fixtures
from backfill_xg import backfill_league_xg
from config import LEAGUES
from src.understat_client import UnderstatUnavailable
from src.understat_loader import UnderstatDataError


@dataclass
class Summary:
    """
    What the refresh found, for deciding the exit status.

    Attributes:
        failed (int): Leagues that could not be refreshed.
        stale (int): Played matches with a blank result, across checked leagues.
        unchecked (int): Leagues whose stale count could not be established.
    """

    failed: int
    stale: int
    unchecked: int


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

    parser.add_argument(
        "--require-fresh",
        action="store_true",
        help=(
            "Exit non-zero when any played match is missing its result, not only "
            "when a league fails. For a scheduled refresh, where a stale results "
            "download is the failure worth catching: the run is green, the CSVs "
            "are green, and the model trains on a month-old season."
        ),
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
        # None rather than 0 until the fixtures step reports, because a league
        # that failed before that point has not been shown to need no results.
        # Defaulting to 0 would report a league as clean for the one thing we
        # did not manage to check.
        "needs_results": None,
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


def report(rows: list[dict], dry_run: bool) -> Summary:
    """
    Prints the table and returns what it found.

    Parameters:
        rows (list[dict]): One row per league, from refresh_league.
        dry_run (bool): Whether this was a dry run.

    Returns:
        Summary: The failed, stale and unchecked counts, for the exit status.
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
    unchecked = 0

    for row in rows:
        if row["error"]:
            failed += 1

        if row["needs_results"] is None:
            unchecked += 1
            stale_text = "-"
        else:
            stale += row["needs_results"]
            stale_text = str(row["needs_results"])

        line = (
            f"{row['league']:<16}"
            f"{row['added']:>10}"
            f"{row['rescheduled']:>10}"
            f"{row['xg_filled']:>8}"
            f"{stale_text:>8}"
        )

        # The error goes after the counts, not instead of them. A league that
        # added its fixtures and then failed on xG has already written to disk,
        # and a report that hid that behind a bare FAILED would leave the reader
        # guessing whether a rerun was needed or whether anything happened.
        if row["error"]:
            line += f"   FAILED: {row['error']}"

        print(line)

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
    elif not failed and not unchecked:
        print("\nEvery played match on file has its result.")

    if unchecked:
        total = len(rows)

        print(
            f"\nThe stale count is unknown for {unchecked} of {total} "
            f"{'league' if total == 1 else 'leagues'}, shown as '-', because "
            "the refresh failed before it could count them. A dash is not a zero."
        )

    if failed:
        total = len(rows)

        print(
            f"\n{failed} of {total} "
            f"{'league' if total == 1 else 'leagues'} could not be refreshed. "
            "Understat publishes no API and no availability, so this is usually "
            "temporary; rerun before trusting the data."
        )

    return Summary(failed=failed, stale=stale, unchecked=unchecked)


def main() -> int:
    """
    Runs the refresh.

    Returns:
        int: 0 on success. 1 if a league failed, or if --require-fresh was
            given and any league is stale or was never checked.
    """
    arguments = parse_arguments()

    leagues = list(LEAGUES) if arguments.league == "all" else [arguments.league]

    rows = [
        refresh_league(league, arguments.season, arguments.dry_run)
        for league in leagues
    ]

    summary = report(rows, arguments.dry_run)

    if summary.failed:
        return 1

    if arguments.require_fresh and (summary.stale or summary.unchecked):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
