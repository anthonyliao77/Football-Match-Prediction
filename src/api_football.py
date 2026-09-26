"""
This module receives data from the API-Football API.
"""

import os
import time

from pathlib import Path

import pandas as pd

# pyrefly: ignore [missing-import]
import requests
from dotenv import load_dotenv

from config import API_FOOTBALL_TEAM_MAP, LEAGUES
from src.data_loader import write_csv_atomic

load_dotenv()

API_BASE_URL = "https://v3.football.api-sports.io"

# Environment variables that may hold the API-Football key, in priority order.
API_KEY_ENV_VARS = ("API_FOOTBALL_KEY", "API_FOOTBALL_API_KEY")

# Match statuses used to select completed fixtures
# (FT: full time, AET: after extra time, PEN: after penalties)
COMPLETED_STATUSES = "FT-AET-PEN"

# The ids parameter of the fixtures endpoint accepts at most 20 IDs per call
MAX_IDS_PER_REQUEST = 20

# Number of attempts made for a request that fails with a retryable status.
MAX_ATTEMPTS = 4

# Base delay in seconds for the exponential backoff between retries.
RETRY_BACKOFF_SECONDS = 1.0

# Columns produced by transform_fixtures, in output order. Referee is added
# only for leagues whose CSVs already carry that column.
FIXTURE_COLUMNS = [
    "FixtureID",
    "Div",
    "Date",
    "Time",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HTHG",
    "HTAG",
    "HTR",
    "Referee",
    "HS",
    "AS",
    "HST",
    "AST",
]

# League directory names mapped to their API-Football league ID.
LEAGUE_IDS = {
    league: config["api_football_id"]
    for league, config in LEAGUES.items()
    if "api_football_id" in config
}

# League directory names whose CSVs carry a Referee column.
REFEREE_LEAGUES = {
    league
    for league, config in LEAGUES.items()
    if config.get("writes_referee")
}


class APIFootballError(RuntimeError):
    """Raised when the API-Football API returns an error or invalid response."""


def _require_api_key() -> str:
    """
    Returns the API-Football API key, validating that it is configured.

    The key is resolved on every call rather than at import time, so that
    importing this module never requires credentials. That keeps the module
    importable for inspection and testing on machines with no API key.

    Returns:
        str: The API-Football API key.

    Raises:
        APIFootballError: If no API key environment variable is set.
    """
    for variable in API_KEY_ENV_VARS:
        api_key = os.getenv(variable)

        if api_key:
            return api_key

    expected = " or ".join(API_KEY_ENV_VARS)

    raise APIFootballError(
        f"{expected} is not configured. Set it in your .env file "
        "or the environment before using the API-Football API."
    )


def _headers() -> dict:
    """
    Builds the request headers for an API-Football call.

    Returns:
        dict: The request headers including the API key.
    """
    return {"x-apisports-key": _require_api_key()}


# A single session is reused across calls so that the batched fixture detail
# requests reuse one connection instead of repeating the TCP and TLS handshake.
_session = requests.Session()


def _quota_remaining(response) -> int | None:
    """
    Reads the remaining request quota from the response headers.

    Parameters:
        response: The requests.Response object to inspect.

    Returns:
        int | None: The remaining quota, or None if the header is absent.
    """
    value = response.headers.get("x-ratelimit-requests-remaining")

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get(endpoint: str, params: dict | None = None) -> dict:
    """
    Sends a GET request to the API-Football API and validates the response.

    Transient failures (429 and 5xx responses, plus connection errors) are
    retried with exponential backoff, because a long backfill issues hundreds
    of requests and a single blip would otherwise abort the whole run.

    Parameters:
        endpoint (str): The API endpoint to call (e.g., "fixtures").
        params (dict | None): Query parameters for the request.

    Returns:
        dict: The parsed JSON response body.

    Raises:
        APIFootballError: If the request fails or the API returns errors.
    """
    url = f"{API_BASE_URL}/{endpoint}"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = _session.get(
                url,
                params=params,
                headers=_headers(),
                timeout=30
            )
        except requests.RequestException as error:
            if attempt == MAX_ATTEMPTS:
                raise APIFootballError(
                    f"Request to /{endpoint} failed: {error}"
                ) from error

            time.sleep(RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1))
            continue

        retryable = (
            response.status_code == 429
            or response.status_code >= 500
        )

        if retryable and attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1))
            continue

        remaining = _quota_remaining(response)

        if remaining is not None:
            print(
                f"API-Football /{endpoint}: {remaining} requests remaining"
            )

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

    raise APIFootballError(f"Request to /{endpoint} failed after retries")


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


def _map_team_name(team: str | None, league: str | None) -> str | None:
    """
    Translates an API-Football team name into its football-data.co.uk spelling.

    Names that are already identical pass through unchanged, so the mapping
    only needs to cover the teams that actually differ between the two sources.

    Parameters:
        team (str | None): The team name as returned by the API.
        league (str | None): The league directory name (e.g. "PremierLeague").

    Returns:
        str | None: The team name in football-data.co.uk spelling.
    """
    if team is None or (not isinstance(team, str) and pd.isna(team)):
        return team

    return API_FOOTBALL_TEAM_MAP.get(league, {}).get(team, team)


def _known_teams(league: str, season: int | str | None = None) -> set:
    """
    Returns the team names already present in a league's CSV files.

    Used to reject API rows for teams the CSV files have never seen, which
    would otherwise be ingested as new clubs and split that team's Elo rating
    and rolling form in two.

    The check spans every season by default. Restricting it to the target
    season would reject a newly promoted club that has not yet played a match
    in that season, even though the club is plainly present in earlier files.

    Parameters:
        league (str): The league directory name (e.g. "PremierLeague").
        season (int | str | None): Restrict to a single season if given.

    Returns:
        set: The team names found in the CSV files.
    """
    directory = Path(LEAGUES[league]["football_data"])

    if season is not None:
        paths = [directory / f"{_csv_season(season)}.csv"]
    else:
        paths = sorted(directory.glob("*.csv"))

    teams = set()

    for path in paths:
        if not path.exists():
            continue

        frame = pd.read_csv(path, usecols=["HomeTeam", "AwayTeam"])
        teams.update(frame["HomeTeam"].dropna().astype(str))
        teams.update(frame["AwayTeam"].dropna().astype(str))

    return teams


def transform_fixtures(
    response: dict,
    statistics: dict | None = None,
    league: str | None = None,
    include_unplayed: bool = False,
) -> pd.DataFrame:
    """
    Transforms the API response into a DataFrame.

    The returned DataFrame always carries the full FIXTURE_COLUMNS schema, so
    an empty API response yields a correctly shaped empty frame rather than a
    zero-column frame.

    Parameters:
        response (dict): The API response containing fixture data.
        statistics (dict | None): Statistics returned by
            fetch_fixture_details, keyed by fixture ID.
        league (str | None): The league directory name (e.g. "PremierLeague").
            Used to translate team names and to select the Div code.
        include_unplayed (bool): Whether to keep fixtures that have no final
            score. Defaults to False, because a row with no score would flow
            into feature computation and Elo as a scoreless match.

    Returns:
        pd.DataFrame: A DataFrame containing fixture information.
    """
    rows = []

    for fixture in response.get("response", []):
        fixture_info = fixture.get("fixture") or {}
        teams = fixture.get("teams") or {}
        goals = fixture.get("goals") or {}
        score = fixture.get("score") or {}
        halftime = score.get("halftime") or {}

        # A fixture with no ID cannot be tracked or deduplicated, and the CSV
        # writer rejects a row without one, so it is dropped here.
        if fixture_info.get("id") is None:
            continue

        home_goals = goals.get("home")
        away_goals = goals.get("away")

        if not include_unplayed and (home_goals is None or away_goals is None):
            continue

        home_halftime = halftime.get("home")
        away_halftime = halftime.get("away")

        row = {
            "FixtureID": fixture_info.get("id"),
            "Div": LEAGUES.get(league, {}).get("div"),
            "Date": _format_date(fixture_info.get("date")),
            "Time": _format_time(fixture_info.get("date")),
            "HomeTeam": _map_team_name(
                (teams.get("home") or {}).get("name"), league
            ),
            "AwayTeam": _map_team_name(
                (teams.get("away") or {}).get("name"), league
            ),
            "FTHG": home_goals,
            "FTAG": away_goals,
            "FTR": _outcome(home_goals, away_goals),
            "HTHG": home_halftime,
            "HTAG": away_halftime,
            "HTR": _outcome(home_halftime, away_halftime),
            "Referee": None,
            "HS": None,
            "AS": None,
            "HST": None,
            "AST": None,
        }

        if league in REFEREE_LEAGUES:
            row["Referee"] = fixture_info.get("referee")

        fixture_statistics = (statistics or {}).get(fixture_info.get("id"))

        if fixture_statistics:
            home_stats = fixture_statistics.get("home") or {}
            away_stats = fixture_statistics.get("away") or {}

            row["HS"] = home_stats.get("Total Shots")
            row["AS"] = away_stats.get("Total Shots")
            row["HST"] = home_stats.get("Shots on Goal")
            row["AST"] = away_stats.get("Shots on Goal")

        rows.append(row)

    columns = [
        column
        for column in FIXTURE_COLUMNS
        if column != "Referee" or league in REFEREE_LEAGUES
    ]

    if league is None:
        columns = FIXTURE_COLUMNS

    return pd.DataFrame(rows, columns=columns)


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


def _date_key(series: pd.Series) -> pd.Series:
    """
    Normalises a date column to a sortable YYYY-MM-DD key for row matching.

    The CSVs store dates as DD/MM/YYYY strings, so the key is parsed with
    dayfirst=True to match how data_loader.py reads them.

    Parameters:
        series (pd.Series): The date column to normalise.

    Returns:
        pd.Series: The dates formatted as YYYY-MM-DD.
    """
    return pd.to_datetime(
        series, dayfirst=True, errors="coerce"
    ).dt.strftime("%Y-%m-%d")


def _merge_key(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the normalised _date_key column used to match API rows to CSV rows.

    Parameters:
        frame (pd.DataFrame): A frame with a Date column.

    Returns:
        pd.DataFrame: A copy of the frame with _date_key added.
    """
    keyed = frame.copy()
    keyed["_date_key"] = _date_key(keyed["Date"])

    return keyed


def _coalesce(frame: pd.DataFrame, columns: list) -> None:
    """
    Fills NaN values in each column from its "_api" suffixed counterpart.

    The unsuffixed column is the existing CSV value and therefore always wins,
    so no football-data.co.uk value is ever overwritten.

    Parameters:
        frame (pd.DataFrame): The merged frame, modified in place.
        columns (list): The column names present in both frames.
    """
    for column in columns:
        api_column = f"{column}_api"

        if api_column in frame.columns:
            frame[column] = frame[column].fillna(frame[api_column])


def _key_tuples(frame: pd.DataFrame) -> pd.Series:
    """
    Builds a comparable tuple key for each row of a frame.

    Parameters:
        frame (pd.DataFrame): A frame with a Date column.

    Returns:
        pd.Series: A (date, home team, away team) tuple per row.
    """
    keyed = _merge_key(frame)

    return pd.Series(
        list(
            zip(
                keyed["_date_key"],
                keyed["HomeTeam"],
                keyed["AwayTeam"],
            )
        ),
        index=keyed.index,
    )


def _augment(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """
    Merges API rows into the existing CSV rows.

    Existing rows that match an API fixture have only their empty columns
    filled in, so no existing value is ever overwritten. Rows with no match
    are appended, but only if they carry a final score, so unplayed fixtures
    never enter the training data.

    Parameters:
        existing (pd.DataFrame): The rows already in the CSV file.
        incoming (pd.DataFrame): The rows produced by transform_fixtures.

    Returns:
        pd.DataFrame: The combined rows, sorted by date then fixture, in the
        original column order with any new columns appended.
    """
    existing = _merge_key(existing)
    incoming = _merge_key(incoming)

    join_on = ["_date_key", "HomeTeam", "AwayTeam"]

    shared = [
        column
        for column in incoming.columns
        if column in existing.columns and column != "FixtureID"
    ]

    extra = [
        column
        for column in incoming.columns
        if column not in existing.columns
    ]

    merged = existing.merge(
        incoming,
        on=join_on,
        how="outer",
        suffixes=("", "_api"),
        indicator=True,
    )

    # The unsuffixed column is the existing CSV value and always wins, so
    # filling only its gaps is what makes this an augment rather than an
    # overwrite. For rows that exist only in the API the unsuffixed column is
    # empty, so the API value is taken instead.
    _coalesce(merged, shared)

    merged = merged.drop(
        columns=[
            f"{column}_api"
            for column in shared
            if f"{column}_api" in merged.columns
        ]
    )

    # A row that exists only in the API is a genuinely new fixture. Keep it
    # only if it has a result, so a scoreless row cannot reach feature
    # computation and be scored as a 0-0 draw.
    is_new = merged["_merge"] == "right_only"
    merged = merged[~is_new | merged["FTR"].notna()]

    merged = merged.sort_values(
        by=["_date_key", "HomeTeam", "AwayTeam"]
    ).drop(columns=["_merge", "_date_key"])

    order = [column for column in existing.columns if column != "_date_key"] + extra

    return merged.reindex(columns=order).reset_index(drop=True)


def _count_filled(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    combined: pd.DataFrame,
) -> int:
    """
    Counts the cells that the merge filled into previously empty CSV cells.

    Only rows present on both sides are considered, and only across the columns
    the two frames share, so neither an appended row nor an existing-only
    column is miscounted.

    Parameters:
        existing (pd.DataFrame): The rows before the merge.
        incoming (pd.DataFrame): The rows produced by transform_fixtures.
        combined (pd.DataFrame): The rows after the merge.

    Returns:
        int: The number of cells filled.
    """
    fillable = [
        column
        for column in incoming.columns
        if column in existing.columns and column != "FixtureID"
    ]

    if not fillable:
        return 0

    existing_keys = _key_tuples(existing)
    incoming_keys = set(_key_tuples(incoming))

    before = _merge_key(existing)
    after = _merge_key(combined)

    matched_before = existing_keys.isin(incoming_keys)
    matched_after = _key_tuples(combined).isin(set(existing_keys))

    gaps_before = int(
        before.loc[matched_before, fillable].isna().sum().sum()
    )
    gaps_after = int(
        after.loc[matched_after, fillable].isna().sum().sum()
    )

    return max(gaps_before - gaps_after, 0)


def _rows_touched(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    combined: pd.DataFrame,
) -> int:
    """
    Counts the matched rows that gained at least one filled cell.

    Where _count_filled answers "how many cells", this answers "how many rows",
    so the update summary can report an "unchanged" row count that means the
    same unit. Comparison is done per merge key rather than by position,
    because the merged frame is reindexed and its row order need not match the
    original.

    Parameters:
        existing (pd.DataFrame): The rows before the merge.
        incoming (pd.DataFrame): The rows produced by transform_fixtures.
        combined (pd.DataFrame): The rows after the merge.

    Returns:
        int: The number of rows that had a gap filled.
    """
    fillable = [
        column
        for column in incoming.columns
        if column in existing.columns and column != "FixtureID"
    ]

    if not fillable:
        return 0

    before = _merge_key(existing)
    after = _merge_key(combined)

    had_gap_before = dict(
        zip(_key_tuples(existing), before[fillable].isna().any(axis=1))
    )
    has_gap_after = dict(
        zip(_key_tuples(combined), after[fillable].isna().any(axis=1))
    )

    return sum(
        1
        for key, gap in had_gap_before.items()
        if gap and not has_gap_after.get(key, True)
    )


def update_csv_file(
    league: str,
    season: int | str,
    new_data: pd.DataFrame,
    dry_run: bool = False,
) -> dict:
    """
    Augments the CSV file for a given league and season with new fixture data.

    API rows are matched to existing rows on date, home team and away team. A
    matched row has only its empty columns filled in, so existing
    football-data.co.uk values (including every odds column) are preserved. An
    unmatched row is appended, provided it has a final score. Running this
    twice with the same input leaves the file unchanged.

    Parameters:
        league (str): The name of the league.
        season (int | str): The season (e.g. 2026 or "2026-2027").
        new_data (pd.DataFrame): A DataFrame containing new fixture data.
        dry_run (bool): If True, report the changes without writing the file.

    Returns:
        dict: Counts of the cells filled in, rows appended and rows left
        unchanged.

    Raises:
        ValueError: If new_data is malformed.
        APIFootballError: If new_data references teams the CSVs have never
            seen, which would silently create duplicate clubs.
    """
    csv_path = (
        Path(LEAGUES[league]["football_data"]) / f"{_csv_season(season)}.csv"
    )

    summary = {"filled": 0, "appended": 0, "unchanged": 0}

    if new_data.empty:
        return summary

    if "FixtureID" not in new_data.columns:
        raise ValueError("new_data is missing the required FixtureID column")

    if new_data["FixtureID"].isnull().any():
        raise ValueError("new_data contains rows without a FixtureID")

    known = _known_teams(league)
    incoming = (
        set(new_data["HomeTeam"].dropna().astype(str))
        | set(new_data["AwayTeam"].dropna().astype(str))
    )

    # Only validate when there is a reference list to validate against. A
    # brand new season file has no known teams yet.
    unknown = sorted(incoming - known) if known else []

    if unknown:
        raise APIFootballError(
            f"API-Football returned teams that are not present in the {league} "
            f"CSVs: {unknown}. Add them to config.API_FOOTBALL_TEAM_MAP before "
            f"ingesting, otherwise that team's Elo rating and rolling form "
            f"would be split in two."
        )

    existing = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()

    if existing.empty:
        combined = new_data.copy()
        summary["appended"] = len(combined)
    else:
        combined = _augment(existing, new_data)

        summary["filled"] = _count_filled(existing, new_data, combined)
        summary["appended"] = len(combined) - len(existing)

        # Counted in rows, not cells, so the three numbers describe the same
        # unit. "Unchanged" means a row that was already present and gained no
        # new information, which is a row that matched an API fixture and had
        # no empty cells filled.
        existing_keys = set(_key_tuples(existing))
        matched = _key_tuples(combined).isin(existing_keys)
        summary["unchanged"] = int(
            matched.sum() - _rows_touched(existing, new_data, combined)
        )

    print(
        f"{'Would update' if dry_run else 'Updated'} {csv_path}: "
        f"{summary['filled']} cells filled, {summary['appended']} rows appended, "
        f"{summary['unchanged']} rows unchanged"
    )

    if dry_run:
        return summary

    write_csv_atomic(csv_path, combined)

    return summary



def _resolve_league(league_id: int, league: str | None) -> str:
    """
    Determines the league directory name from a league ID.

    Parameters:
        league_id (int): The API-Football league ID.
        league (str | None): The league directory name, if already known.

    Returns:
        str: The league directory name.

    Raises:
        APIFootballError: If the league ID is not in LEAGUE_IDS.
    """
    if league is not None:
        return league

    resolved = next(
        (name for name, lid in LEAGUE_IDS.items() if lid == league_id),
        None
    )

    if resolved is None:
        raise APIFootballError(
            f"Unknown league_id {league_id}. Pass the league name or add it "
            "to config.LEAGUES."
        )

    return resolved


def update_league_data(
    league_id: int,
    season: int | str,
    league: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    dry_run: bool = False,
) -> dict:
    """
    Incrementally updates the league data with recently completed fixtures.

    For daily use, pass a small ``from_date``/``to_date`` window (for example
    yesterday to today) so that only newly completed fixtures are fetched
    instead of downloading the entire season again. Re-fetching fixtures that
    are already in the CSV is safe: rows are matched on date and teams, only
    empty cells are filled, and rows that are already present are left alone.

    Parameters:
        league_id (int): The ID of the league.
        season (int | str): The season year (e.g., 2024).
        league (str | None): The league directory name (e.g., "PremierLeague").
            Inferred from the league ID when not provided.
        from_date (str | None): Start date (YYYY-MM-DD) for the update window.
        to_date (str | None): End date (YYYY-MM-DD) for the update window.
        dry_run (bool): If True, report the changes without writing the file.

    Returns:
        dict: Counts of the cells filled in, rows appended and rows unchanged.
    """
    league = _resolve_league(league_id, league)

    completed = fetch_completed_fixtures(
        league_id,
        season,
        from_date=from_date,
        to_date=to_date
    )

    fixtures = completed.get("response", [])

    if not fixtures:
        print(
            f"No completed {league} fixtures returned for season {season} "
            f"in the requested window. Nothing to do."
        )

        return {"filled": 0, "appended": 0, "unchanged": 0}

    fixture_ids = [
        (fixture.get("fixture") or {}).get("id")
        for fixture in fixtures
    ]
    fixture_ids = [
        fixture_id for fixture_id in fixture_ids if fixture_id is not None
    ]

    # Statistics are a bonus on top of the results. Some plans do not allow the
    # ids parameter, and the fixtures endpoint does not return statistics
    # inline, so a rejection here must not cost us the scores.
    try:
        statistics = fetch_fixture_details(fixture_ids)
    except APIFootballError as error:
        print(
            f"Could not fetch match statistics: {error}\n"
            f"Continuing without shots data. Scores and results will still "
            f"be ingested, and any empty HS/AS/HST/AST cells in the CSV will "
            f"be left as they are."
        )
        statistics = {}

    new_data = transform_fixtures(completed, statistics, league=league)

    return update_csv_file(league, season, new_data, dry_run=dry_run)


def create_initial_season_data(
    league_id: int,
    season: int | str,
    league: str | None = None,
    include_unplayed: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Creates the season CSV from the full fixture schedule.

    Only completed fixtures are written by default. A fixture with no final
    score has no result to learn from, and because the feature builder and the
    Elo walk both iterate every row in the CSV, writing one would add a
    spurious 0-0 draw to that team's record. Pass ``include_unplayed=True`` to
    write the whole schedule regardless, for example to inspect the fixture
    list.

    Parameters:
        league_id (int): The ID of the league.
        season (int | str): The season year (e.g., 2024).
        league (str | None): The league directory name (e.g., "PremierLeague").
            Inferred from the league ID when not provided.
        include_unplayed (bool): Whether to write fixtures that have no final
            score. Defaults to False.
        dry_run (bool): If True, report the changes without writing the file.

    Returns:
        dict: Counts of the cells filled in, rows appended and rows unchanged.
    """
    league = _resolve_league(league_id, league)

    schedule = fetch_league_schedule(league_id, season)

    if not schedule.get("response"):
        print(
            f"No {league} fixtures returned for season {season}. "
            f"Nothing to do."
        )

        return {"filled": 0, "appended": 0, "unchanged": 0}

    new_data = transform_fixtures(
        schedule,
        league=league,
        include_unplayed=include_unplayed,
    )

    return update_csv_file(league, season, new_data, dry_run=dry_run)
