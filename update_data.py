"""
Augments the local football-data.co.uk CSVs with API-Football fixtures.
"""

import argparse

from config import LEAGUES
from src.api_football import create_initial_season_data, update_league_data


def parse_arguments():
    """
    Parse command-line arguments for the data update script.
    """
    parser = argparse.ArgumentParser(
        description="Augment local league CSVs with API-Football fixtures."
    )

    parser.add_argument(
        "--league",
        required=True,
        choices=sorted(LEAGUES),
        help="League to update."
    )

    parser.add_argument(
        "--season",
        required=True,
        help="Season to update, as a start year or a range "
             "(e.g. 2026 or 2026-2027)."
    )

    parser.add_argument(
        "--from",
        dest="from_date",
        help="Start date of the update window (YYYY-MM-DD). Restricting the "
             "window keeps the request count down."
    )

    parser.add_argument(
        "--to",
        dest="to_date",
        help="End date of the update window (YYYY-MM-DD)."
    )

    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Backfill from the full fixture schedule instead of only "
             "completed fixtures."
    )

    parser.add_argument(
        "--include-unplayed",
        action="store_true",
        help="With --schedule, also write fixtures that have no final score. "
             "Off by default, because a scoreless row is scored as a 0-0 draw "
             "by the feature builder and the Elo walk."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing the CSV. The API calls "
             "are still made, so this does not save quota."
    )

    return parser.parse_args()


def main():
    """
    Runs the requested API-Football update.
    """
    args = parse_arguments()

    league_id = LEAGUES[args.league]["api_football_id"]

    if args.schedule:
        summary = create_initial_season_data(
            league_id=league_id,
            season=args.season,
            league=args.league,
            include_unplayed=args.include_unplayed,
            dry_run=args.dry_run,
        )
    else:
        summary = update_league_data(
            league_id=league_id,
            season=args.season,
            league=args.league,
            from_date=args.from_date,
            to_date=args.to_date,
            dry_run=args.dry_run,
        )

    if not args.dry_run:
        print(
            f"Done: {summary['filled']} cells filled, "
            f"{summary['appended']} rows appended."
        )


if __name__ == "__main__":
    main()
