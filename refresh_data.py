"""
Refreshes the season CSVs from Understat in one command.

A season file needs three things kept current. The fixtures still to come and
the xG of the matches already played both come from Understat, and the results
of matches played since the last download do too. sync_understat.py and
backfill_xg.py each know how to get one of the first two, and the first also
fills in results now, so this is an orchestrator rather than new logic. It calls
the two existing entry points, aggregates what they report, and ends by saying
the one thing the person running it needs to know: whether any played match is
still sitting in the file without a result.

That number used to mean something different. Results came from a
football-data.co.uk download, so a match with a blank result meant the download
was behind, and the count was how far behind. Understat publishes a score as
soon as a match is over, so the file fills itself in now and a blank result is
not a chore left undone. It is a match the refresh failed to finish: an
abandoned or unplayed fixture, or a request that did not come back. That is why
the report no longer tells you to go and download anything.

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

A league that failed is counted as having an unknown number of unwritten
matches, not as having none, and the table says so with a dash. The alternative
was a refresh that reported every league as clean while one of them had not been
looked at, which is the failure mode a staleness check exists to prevent.

--require-fresh extends the exit status to cover an unfinished match. A failed
refresh is loud, but the likelier way for a scheduled run to be quietly useless
is for everything to succeed while one match sits unwritten.
"""

import argparse
import sys
from dataclasses import dataclass

from backfill_xg import backfill_league_xg
from config import LEAGUES
from src.understat_client import UnderstatUnavailable
from src.understat_loader import UnderstatDataError
from sync_understat import _default_season, sync_league


@dataclass
class Summary:
    """
    What the refresh found, for deciding the exit status.

    Attributes:
        failed (int): Leagues that could not be refreshed.
        stale (int): Played matches left with a blank result, across
            checked leagues. The name is the old one and stays, since it
            reads as what it is: the season not being current.
        unchecked (int): Leagues whose count could not be established.
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
            "Refresh the fixtures, the results and the xG of every league "
            "from Understat. Existing values on file are never overwritten."
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
            "when a league fails. For a scheduled refresh, where a match left "
            "unwritten is the failure worth catching: the run is green, the CSVs "
            "are green, and the model trains on a season that is quietly short "
            "a game."
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
        "results_filled": 0,
        "results_added": 0,
        "disagreements": 0,
        "match_requests": 0,
        "xg_filled": 0,
        "rows": 0,
        "rows_unmatched": 0,
        "error": None,
    }

    if season is None:
        row["error"] = f"no season CSVs found in {directory}"

        return row

    try:
        fixtures = sync_league(league, season, dry_run=dry_run)

        for key in (
            "added",
            "rescheduled",
            "needs_results",
            "results_filled",
            "results_added",
            "disagreements",
            "match_requests",
        ):
            row[key] = fixtures[key]
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
    print("Unwritten is a played match with no result on file. Results is how")
    print("many were filled this run. An xG count of zero is the normal")
    print("case: it only fills cells that are empty.\n")

    header = (
        f"{'League':<16}{'Fixtures':>10}{'Re-dated':>10}{'Results':>9}"
        f"{'xG':>8}{'Unwritten':>11}"
    )

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
            unwritten = "-"
        else:
            stale += row["needs_results"]
            unwritten = str(row["needs_results"])

        results = (
            row.get("results_filled", 0) + row.get("results_added", 0)
        )

        line = (
            f"{row['league']:<16}"
            f"{row['added']:>10}"
            f"{row['rescheduled']:>10}"
            f"{results:>9}"
            f"{row['xg_filled']:>8}"
            f"{unwritten:>11}"
        )

        # The error goes after the counts, not instead of them. A league that
        # added its fixtures and then failed on xG has already written to disk,
        # and a report that hid that behind a bare FAILED would leave the reader
        # guessing whether a rerun was needed or whether anything happened.
        if row["error"]:
            line += f"   FAILED: {row['error']}"

        if row.get("disagreements"):
            line += f"   ({row['disagreements']} source disagreement(s))"

        print(line)

    print(
        "\nResults filled from Understat. Unwritten is a played match still "
        "holding no\nresult, which is a fault to look at rather than a "
        "download to make."
    )

    if stale:
        print(
            f"\n{stale} played "
            f"{'match is' if stale == 1 else 'matches are'} still without a "
            f"result in the CSV."
        )
        print(
            "These are the ones the refresh could not finish: an abandoned or "
            "unplayed\nfixture, a match with no shots to read, or a request "
            "that failed. Each is\nlisted by the fixtures step."
        )
    elif not failed and not unchecked:
        print("\nEvery played match on file has its result.")

    if unchecked:
        total = len(rows)

        print(
            f"\nThe count is unknown for {unchecked} of {total} "
            f"{'league' if total == 1 else 'leagues'}, shown as '-', because the "
            "refresh failed before it could count them. A dash is not a zero."
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
