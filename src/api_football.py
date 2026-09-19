"""
This module receives data from the API-Football API.
"""

import os

from pathlib import Path

import pandas as pd

# pyrefly: ignore [missing-import]
import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = "https://v3.football.api-sports.io"

# Match statuses used to select completed fixtures
# (FT: full time, AET: after extra time, PEN: after penalties)
COMPLETED_STATUSES = "FT-AET-PEN"

# The ids parameter of the fixtures endpoint accepts at most 20 IDs per call
MAX_IDS_PER_REQUEST = 20

# Mapping of league names used for CSV directories to API-Football league IDs
LEAGUE_IDS = {
    "PremierLeague": 39,
    "LaLiga": 140,
    "SerieA": 135,
}


class APIFootballError(RuntimeError):
    """Raised when the API-Football API returns an error or invalid response."""


def _require_api_key() -> str:
    """
    Returns the API-Football API key, validating that it is configured.

    Returns:
        str: The API-Football API key.

    Raises:
        APIFootballError: If the API_FOOTBALL_KEY environment variable is
            not set.
    """
    api_key = os.getenv("API_FOOTBALL_KEY")

    if not api_key:
        raise APIFootballError(
            "API_FOOTBALL_KEY is not configured. Set it in your .env file "
            "or the environment before using the API-Football API."
        )

    return api_key


API_KEY = _require_api_key()

headers = {
    "x-apisports-key": API_KEY
}


def _get(endpoint: str, params: dict | None = None) -> dict:
    """
    Sends a GET request to the API-Football API and validates the response.

    Parameters:
        endpoint (str): The API endpoint to call (e.g., "fixtures").
        params (dict | None): Query parameters for the request.

    Returns:
        dict: The parsed JSON response body.

    Raises:
        APIFootballError: If the request fails or the API returns errors.
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/{endpoint}",
            params=params,
            headers=headers,
            timeout=30
        )
    except requests.RequestException as error:
        raise APIFootballError(
            f"Request to /{endpoint} failed: {error}"
        ) from error

    if response.status_code != 200:
        raise APIFootballError(
            f"API-Football returned HTTP {response.status_code} "
            f"for /{endpoint}"
        )

    try:
        data = response.json()
    except ValueError as error:
        raise APIFootballError(
            f"Invalid JSON response from /{endpoint}"
        ) from error

    if data.get("errors"):
        raise APIFootballError(f"API-Football errors: {data['errors']}")

    return data


def _api_season(season: int | str) -> int:
    """
    Normalises a season value to the 4-digit start year used by the API.

    Parameters:
        season (int | str): The season (e.g., 2024, "2024-2025" or "2024/2025").

    Returns:
        int: The season start year.
    """
    if isinstance(season, str):
        for separator in ("-", "/"):
            if separator in season:
                return int(season.split(separator)[0])

    return int(season)


def _csv_season(season: int | str) -> str:
    """
    Converts a season value to the YYYY-YYYY format used in CSV filenames.

    Parameters:
        season (int | str): The season (e.g., 2026, "2026-2027" or
            "2026/2027").

    Return:
        str: The season in YYYY-YYYY format.
    """
    start = _api_season(season)

    return f"{start}-{start + 1}"


def _chunk_ids(fixture_ids: list[int], size: int) -> list[list[int]]:
    """
    Splits a list into consecutive chunks of at most ``size`` items.

    Parameters:
        fixture_ids (list[int]): The list of fixture IDs.
        size (int): The maximum chunk size.

    Returns:
        list[list[int]]: The list split into chunks.
    """
    return [
        fixture_ids[start:start + size]
        for start in range(0, len(fixture_ids), size)
    ]


def _outcome(home_goals: int | None, away_goals: int | None) -> str | None:
    """
    Derives a match outcome (H, D or A) from the two goals totals.

    Parameters:
        home_goals (int | None): Goals scored by the home team.
        away_goals (int | None): Goals scored by the away team.

    Returns:
        str | None: "H", "D" or "A", or None if the score is unknown.
    """
    if home_goals is None or away_goals is None:
        return None

    if home_goals == away_goals:
        return "D"

    return "H" if home_goals > away_goals else "A"


def _fetch_fixtures(params: dict) -> dict:
    """
    Fetches all fixtures matching the given parameters.

    The fixtures endpoint returns every matching fixture in a single
    response and does not accept a ``page`` parameter, so no pagination
    argument is sent.

    Parameters:
        params (dict): The query parameters for the request.

    Returns:
        dict: The raw API response containing the fixtures.
    """
    return _get("fixtures", params=params)


def fetch_league_schedule(
    league_id: int,
    season: int | str,
    from_date: str | None = None,
    to_date: str | None = None,
) -> dict:
    """
    Fetches the full season fixture schedule for a given league and season,
    including fixtures that have not been played yet.

    Parameters:
        league_id (int): The ID of the league.
        season (int | str): The season year (e.g., 2024).
        from_date (str | None): Start date (YYYY-MM-DD) to fetch only the
            fixtures of that period.
        to_date (str | None): End date (YYYY-MM-DD) to fetch only the
            fixtures of that period.

    Returns:
        dict: A dictionary containing the league schedule.
    """
    params = {
        "league": league_id,
        "season": _api_season(season),
    }

    if from_date is not None:
        params["from"] = from_date

    if to_date is not None:
        params["to"] = to_date

    return _fetch_fixtures(params)


def fetch_completed_fixtures(
    league_id: int,
    season: int | str,
    from_date: str | None = None,
    to_date: str | None = None,
) -> dict:
    """
    Fetches the completed fixtures for a given league and season.

    Only fixtures with a completed status (FT, AET or PEN) are returned,
    optionally restricted to a date range for incremental updates.

    Parameters:
        league_id (int): The ID of the league.
        season (int | str): The season year (e.g., 2024).
        from_date (str | None): Start date (YYYY-MM-DD) to fetch only the
            fixtures of that period.
        to_date (str | None): End date (YYYY-MM-DD) to fetch only the
            fixtures of that period.

    Returns:
        dict: A dictionary containing the completed fixtures.
    """
    params = {
        "league": league_id,
        "season": _api_season(season),
        "status": COMPLETED_STATUSES,
    }

    if from_date is not None:
        params["from"] = from_date

    if to_date is not None:
        params["to"] = to_date

    return _fetch_fixtures(params)


def fetch_fixture_details(fixture_ids: list[int]) -> dict:
    """
    Fetches detailed statistics for a batch of fixtures.

    The fixtures endpoint accepts a maximum of ``MAX_IDS_PER_REQUEST`` fixture
    IDs per call and returns the match statistics for each of them, so the
    batch is fetched with as few requests as possible.

    Parameters:
        fixture_ids (list[int]): A list of fixture IDs.

    Returns:
        dict: A dictionary containing detailed fixture statistics,
            keyed by fixture ID with "home" and "away" team statistic maps.
    """
    details = {}

    for chunk in _chunk_ids(fixture_ids, MAX_IDS_PER_REQUEST):
        params = {"ids": "-".join(str(fixture_id) for fixture_id in chunk)}

        data = _get("fixtures", params=params)

        for fixture in data.get("response", []):
            fixture_info = fixture.get("fixture", {})

            fixture_id = fixture_info.get("id")

            if fixture_id is None:
                continue

            teams = fixture.get("teams", {})

            home_team_id = teams.get("home", {}).get("id")
            away_team_id = teams.get("away", {}).get("id")

            team_statistics = {"home": {}, "away": {}}

            for team in fixture.get("statistics", []):
                team_id = team.get("team", {}).get("id")

                if team_id == home_team_id:
                    side = "home"
                elif team_id == away_team_id:
                    side = "away"
                else:
                    continue

                for stat in team.get("statistics", []):
                    team_statistics[side][stat.get("type")] = stat.get("value")

            details[fixture_id] = team_statistics

    return details


def transform_fixtures(
    response: dict,
    statistics: dict | None = None
) -> pd.DataFrame:
    """
    Transforms the API response into a DataFrame.

    Environment variables:
        None

    Parameters:
        response (dict): The API response containing fixture data.
        statistics (dict | None): Statistics returned by
            fetch_fixture_details, keyed by fixture ID.

    Returns:
        pd.DataFrame: A DataFrame containing fixture information.
    """
    rows = []

    for fixture in response.get("response", []):
        fixture_info = fixture.get("fixture", {})
        league_info = fixture.get("league", {})
        teams = fixture.get("teams", {})
        goals = fixture.get("goals", {})
        score = fixture.get("score", {})
        halftime = score.get("halftime", {})

        home_goals = goals.get("home")
        away_goals = goals.get("away")

        home_halftime = halftime.get("home")
        away_halftime = halftime.get("away")

        row = {
            "FixtureID": fixture_info.get("id"),
            "Div": league_info.get("name"),
            "Date": _format_date(fixture_info.get("date")),
            "Time": _format_time(fixture_info.get("date")),
            "HomeTeam": teams.get("home", {}).get("name"),
            "AwayTeam": teams.get("away", {}).get("name"),
            "FTHG": home_goals,
            "FTAG": away_goals,
            "FTR": _outcome(home_goals, away_goals),
            "HTHG": home_halftime,
            "HTAG": away_halftime,
            "HTR": _outcome(home_halftime, away_halftime),
            "Referee": fixture_info.get("referee"),
            "HS": None,
            "AS": None,
            "HST": None,
            "AST": None,
        }

        fixture_statistics = (statistics or {}).get(fixture_info.get("id"))

        if fixture_statistics:
            home_stats = fixture_statistics.get("home", {})
            away_stats = fixture_statistics.get("away", {})

            row["HS"] = home_stats.get("Total Shots")
            row["AS"] = away_stats.get("Total Shots")
            row["HST"] = home_stats.get("Shots on Goal")
            row["AST"] = away_stats.get("Shots on Goal")

        rows.append(row)

    return pd.DataFrame(rows)


def _format_date(iso_date: str | None) -> str | None:
    """
    Formats an ISO date string into the DD/MM/YYYY format used by the dataset.

    The API returns kickoff times in UTC by default (offset +00:00), and that
    UTC calendar date is stored as-is. Timezone conversion can be done by
    passing a ``timezone`` parameter to the fetch functions; it is not
    applied here.

    Parameters:
        iso_date (str | None): An ISO 8601 date string.

    Returns:
        str | None: The formatted kickoff date (UTC), or None when missing.
    """
    if iso_date is None:
        return None

    return pd.to_datetime(iso_date).tz_localize(None).strftime("%d/%m/%Y")


def _format_time(iso_date: str | None) -> str | None:
    """
    Extracts the HH:MM kickoff time from an ISO date string.

    The API returns kickoff times in UTC by default, and that UTC time is
    stored as-is. Timezone conversion can be done by passing a ``timezone``
    parameter to the fetch functions; it is not applied here.

    Parameters:
        iso_date (str | None): An ISO 8601 date string.

    Returns:
        str | None: The kickoff time (UTC), or None when missing.
    """
    if iso_date is None:
        return None

    return pd.to_datetime(iso_date).tz_localize(None).strftime("%H:%M")


def update_csv_file(league: str, season: int | str, new_data: pd.DataFrame) -> None:
    """
    Updates the CSV file for a given league and season with new fixture data.

    Parameters:
        league (str): The name of the league.
        season (int | str): The season (e.g., 2026 or "2026-2027").
        new_data (pd.DataFrame): A DataFrame containing new fixture data.
    """
    csv_path = Path(f"football_data/{league}/{_csv_season(season)}.csv")

    if "FixtureID" not in new_data.columns:
        raise ValueError("new_data is missing the required FixtureID column")

    if new_data["FixtureID"].isnull().any():
        raise ValueError("new_data contains rows without a FixtureID")

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        existing_data = pd.read_csv(csv_path)

        combined_data = pd.concat(
            [existing_data, new_data],
            ignore_index=True
        )

        combined_data = combined_data.drop_duplicates(
            subset="FixtureID",
            keep="last"
        )
    else:
        combined_data = new_data

    combined_data.to_csv(csv_path, index=False)


def update_league_data(
    league_id: int,
    season: int | str,
    league: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
) -> None:
    """
    Incrementally updates the league data with recently completed fixtures.

    For daily use, pass a small ``from_date``/``to_date`` window (for
    example yesterday to today) so that only newly completed fixtures are
    fetched instead of downloading the entire season again. Re-fetching
    fixtures that are already in the CSV is safe: rows are deduplicated by
    FixtureID and the newest version is kept.

    Parameters:
        league_id (int): The ID of the league.
        season (int | str): The season year (e.g., 2024).
        league (str | None): The league directory name (e.g., "PremierLeague").
            Inferred from the league ID when not provided.
        from_date (str | None): Start date (YYYY-MM-DD) for the update window.
        to_date (str | None): End date (YYYY-MM-DD) for the update window.
    """
    if league is None:
        league = next(
            (name for name, lid in LEAGUE_IDS.items() if lid == league_id),
            None
        )

    if league is None:
        raise APIFootballError(
            f"Unknown league_id {league_id}. Pass the league name or add it "
            "to LEAGUE_IDS."
        )

    completed = fetch_completed_fixtures(
        league_id,
        season,
        from_date=from_date,
        to_date=to_date
    )

    fixtures = completed.get("response", [])

    fixture_ids = [
        fixture["fixture"]["id"]
        for fixture in fixtures
        if (fixture.get("fixture") or {}).get("id") is not None
    ]

    statistics = fetch_fixture_details(fixture_ids)

    new_data = transform_fixtures(completed, statistics)

    update_csv_file(league, season, new_data)


def create_initial_season_data(
    league_id: int,
    season: int | str,
    league: str | None = None,
) -> None:
    """
    Creates the season CSV from the full fixture schedule.

    The schedule includes future fixtures, so no detailed statistics are
    fetched here. Completed fixtures are written with statistics as they
    become available through update_league_data.

    Parameters:
        league_id (int): The ID of the league.
        season (int | str): The season year (e.g., 2024).
        league (str | None): The league directory name (e.g., "PremierLeague").
            Inferred from the league ID when not provided.
    """
    if league is None:
        league = next(
            (name for name, lid in LEAGUE_IDS.items() if lid == league_id),
            None
        )

    if league is None:
        raise APIFootballError(
            f"Unknown league_id {league_id}. Pass the league name or add it "
            "to LEAGUE_IDS."
        )

    schedule = fetch_league_schedule(league_id, season)

    new_data = transform_fixtures(schedule)

    update_csv_file(league, season, new_data)