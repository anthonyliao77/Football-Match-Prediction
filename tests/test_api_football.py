"""
Tests for the API-Football module.
"""

import pandas as pd
import pytest

from src import api_football


class MockResponse:
    """Minimal stand-in for a requests.Response object."""

    def __init__(self, data, status_code=200, headers=None):
        self.data = data
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self.data


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    """
    Sets a placeholder key for every test.

    The key is resolved per call rather than at import time, so this only
    needs to be present for the tests that actually issue a request.
    """
    monkeypatch.setenv("API_FOOTBALL_KEY", "test_api_key")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Keeps the retry backoff from slowing the test suite down."""
    monkeypatch.setattr(api_football.time, "sleep", lambda seconds: None)


def mock_session_get(monkeypatch, handler):
    """
    Patches the module's session with a request handler.
    """
    monkeypatch.setattr(api_football._session, "get", handler)


@pytest.fixture
def fixture_response():
    """
    A single completed fixture in the API-Football response format.
    """
    return {
        "fixture": {
            "id": 1208021,
            "referee": "M. Oliver",
            "date": "2024-08-16T19:00:00+00:00",
        },
        "league": {
            "id": 39,
            "name": "Premier League",
            "season": 2024,
        },
        "teams": {
            "home": {"id": 33, "name": "Man United"},
            "away": {"id": 34, "name": "Fulham"},
        },
        "goals": {"home": 1, "away": 0},
        "score": {
            "halftime": {"home": 0, "away": 0},
        },
    }


@pytest.fixture
def unplayed_response():
    """
    A scheduled fixture that has not been played yet.
    """
    return {
        "fixture": {
            "id": 1208022,
            "referee": "A. Mariner",
            "date": "2024-12-20T19:00:00+00:00",
        },
        "league": {
            "id": 39,
            "name": "Premier League",
            "season": 2024,
        },
        "teams": {
            "home": {"id": 33, "name": "Man United"},
            "away": {"id": 34, "name": "Fulham"},
        },
        "goals": {"home": None, "away": None},
        "score": {"halftime": {"home": None, "away": None}},
    }


def write_csv(directory, league, season, rows, columns=None):
    """
    Writes a CSV shaped like the football-data.co.uk files.
    """
    path = directory / "football_data" / league / f"{season}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(rows)

    if columns:
        for column in columns:
            if column not in frame.columns:
                frame[column] = None

    frame.to_csv(path, index=False)

    return path


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------


def test_module_imports_without_api_key():
    """
    Tests that importing the module does not require an API key.

    Resolving the key at import time meant the module could not be inspected,
    tested or imported at all on a machine without credentials.
    """
    assert api_football.API_KEY_ENV_VARS[0] == "API_FOOTBALL_KEY"


def test_require_api_key_reads_canonical_variable(monkeypatch):
    """Tests that API_FOOTBALL_KEY is used when it is set."""
    monkeypatch.setenv("API_FOOTBALL_KEY", "canonical")

    assert api_football._require_api_key() == "canonical"


def test_require_api_key_falls_back_to_legacy_variable(monkeypatch):
    """
    Tests that the legacy API_FOOTBALL_API_KEY name is still accepted.

    The name in .env did not match the name the module read, which made the
    key invisible to the code.
    """
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    monkeypatch.setenv("API_FOOTBALL_API_KEY", "legacy")

    assert api_football._require_api_key() == "legacy"


def test_missing_api_key_raises(monkeypatch):
    """
    Tests that a request without a configured key raises a clear error.
    """
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    monkeypatch.delenv("API_FOOTBALL_API_KEY", raising=False)

    with pytest.raises(api_football.APIFootballError, match="not configured"):
        api_football._headers()


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def test_get_sends_the_api_key_header(monkeypatch):
    """Tests that the key is sent as the x-apisports-key header."""
    captured = {}

    def mock_get(url, params=None, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers

        return MockResponse({"response": []})

    mock_session_get(monkeypatch, mock_get)

    api_football._get("fixtures", {"league": 39})

    assert captured["url"].endswith("/fixtures")
    assert captured["headers"]["x-apisports-key"] == "test_api_key"


def test_get_raises_on_api_error(monkeypatch):
    """Tests that an error payload raises APIFootballError."""

    def mock_get(url, params=None, headers=None, timeout=None):
        return MockResponse({"errors": {"plan": "no access"}})

    mock_session_get(monkeypatch, mock_get)

    with pytest.raises(api_football.APIFootballError, match="no access"):
        api_football._get("fixtures")


def test_get_raises_on_http_error(monkeypatch):
    """Tests that a non-200 status raises APIFootballError."""

    def mock_get(url, params=None, headers=None, timeout=None):
        return MockResponse({}, status_code=400)

    mock_session_get(monkeypatch, mock_get)

    with pytest.raises(api_football.APIFootballError, match="HTTP 400"):
        api_football._get("fixtures")


def test_get_retries_on_rate_limit(monkeypatch):
    """
    Tests that a 429 is retried and then succeeds.

    A backfill issues hundreds of requests, so a single throttled response must
    not abandon the run.
    """
    calls = {"count": 0}

    def mock_get(url, params=None, headers=None, timeout=None):
        calls["count"] += 1

        if calls["count"] == 1:
            return MockResponse({}, status_code=429)

        return MockResponse({"response": [{"ok": True}]})

    mock_session_get(monkeypatch, mock_get)

    data = api_football._get("fixtures")

    assert calls["count"] == 2
    assert data["response"] == [{"ok": True}]


def test_get_retries_on_server_error(monkeypatch):
    """Tests that a 500 is retried."""
    calls = {"count": 0}

    def mock_get(url, params=None, headers=None, timeout=None):
        calls["count"] += 1

        if calls["count"] < 3:
            return MockResponse({}, status_code=503)

        return MockResponse({"response": []})

    mock_session_get(monkeypatch, mock_get)

    api_football._get("fixtures")

    assert calls["count"] == 3


def test_get_gives_up_after_max_attempts(monkeypatch):
    """Tests that retries are bounded rather than endless."""
    calls = {"count": 0}

    def mock_get(url, params=None, headers=None, timeout=None):
        calls["count"] += 1

        return MockResponse({}, status_code=429)

    mock_session_get(monkeypatch, mock_get)

    with pytest.raises(api_football.APIFootballError, match="HTTP 429"):
        api_football._get("fixtures")

    assert calls["count"] == api_football.MAX_ATTEMPTS


def test_get_does_not_retry_client_error(monkeypatch):
    """Tests that a 400 is raised immediately rather than retried."""
    calls = {"count": 0}

    def mock_get(url, params=None, headers=None, timeout=None):
        calls["count"] += 1

        return MockResponse({}, status_code=400)

    mock_session_get(monkeypatch, mock_get)

    with pytest.raises(api_football.APIFootballError):
        api_football._get("fixtures")

    assert calls["count"] == 1


def test_quota_remaining_reads_header():
    """Tests that the remaining quota is read from the response headers."""
    response = MockResponse({}, headers={"x-ratelimit-requests-remaining": "97"})

    assert api_football._quota_remaining(response) == 97


def test_quota_remaining_handles_missing_header():
    """Tests that a missing quota header is reported as unknown."""
    assert api_football._quota_remaining(MockResponse({})) is None


# ---------------------------------------------------------------------------
# Season helpers
# ---------------------------------------------------------------------------


def test_api_season_normalises_formats():
    """Tests that a season is reduced to its four digit start year."""
    assert api_football._api_season(2026) == 2026
    assert api_football._api_season("2026") == 2026
    assert api_football._api_season("2026-2027") == 2026
    assert api_football._api_season("2026/2027") == 2026


def test_csv_season_formats():
    """Tests that a season is formatted as YYYY-YYYY for the CSV filename."""
    assert api_football._csv_season(2026) == "2026-2027"
    assert api_football._csv_season("2026-2027") == "2026-2027"


# ---------------------------------------------------------------------------
# Request parameters
# ---------------------------------------------------------------------------


def test_fetch_league_schedule_requests_fixtures(monkeypatch, fixture_response):
    """Tests that fetch_league_schedule requests the full season schedule."""
    captured = {}

    def mock_get(url, params=None, headers=None, timeout=None):
        captured.update(params)

        return MockResponse({"response": [fixture_response]})

    mock_session_get(monkeypatch, mock_get)

    api_football.fetch_league_schedule(39, 2026)

    assert captured["league"] == 39
    assert captured["season"] == 2026
    assert "status" not in captured


def test_fetch_completed_fixtures_filters_by_status(
    monkeypatch,
    fixture_response,
):
    """Tests that only completed fixtures are requested."""
    captured = {}

    def mock_get(url, params=None, headers=None, timeout=None):
        captured.update(params)

        return MockResponse({"response": [fixture_response]})

    mock_session_get(monkeypatch, mock_get)

    api_football.fetch_completed_fixtures(
        39, 2026, from_date="2026-08-16", to_date="2026-08-17"
    )

    assert captured["status"] == api_football.COMPLETED_STATUSES
    assert captured["from"] == "2026-08-16"
    assert captured["to"] == "2026-08-17"


def test_fixtures_requests_do_not_send_page(monkeypatch, fixture_response):
    """
    Tests that no page parameter is sent.

    The fixtures endpoint returns every match in one response and rejects the
    parameter.
    """
    captured = {}

    def mock_get(url, params=None, headers=None, timeout=None):
        captured.update(params)

        return MockResponse({"response": [fixture_response]})

    mock_session_get(monkeypatch, mock_get)

    api_football.fetch_league_schedule(39, 2026)

    assert "page" not in captured


def test_fetch_fixture_details_batches_statistics(monkeypatch):
    """Tests that fixture IDs are requested in batches of twenty."""
    calls = []

    def mock_get(url, params=None, headers=None, timeout=None):
        calls.append(params)

        return MockResponse({
            "response": [
                {
                    "fixture": {"id": int(fixture_id)},
                    "teams": {"home": {"id": 1}, "away": {"id": 2}},
                    "statistics": [
                        {
                            "team": {"id": 1},
                            "statistics": [
                                {"type": "Total Shots", "value": 10}
                            ],
                        },
                        {
                            "team": {"id": 2},
                            "statistics": [
                                {"type": "Total Shots", "value": 5}
                            ],
                        },
                    ],
                }
                for fixture_id in params["ids"].split("-")
            ]
        })

    mock_session_get(monkeypatch, mock_get)

    details = api_football.fetch_fixture_details(list(range(1, 46)))

    assert len(calls) == 3

    for params in calls:
        assert len(params["ids"].split("-")) <= api_football.MAX_IDS_PER_REQUEST

    assert details[1]["home"]["Total Shots"] == 10
    assert details[1]["away"]["Total Shots"] == 5


# ---------------------------------------------------------------------------
# transform_fixtures
# ---------------------------------------------------------------------------


def test_transform_fixtures(fixture_response):
    """Tests that a fixture is transformed into a flat row."""
    result = api_football.transform_fixtures(
        {"response": [fixture_response]},
        league="PremierLeague",
    )

    row = result.iloc[0]

    assert row["FixtureID"] == 1208021
    assert row["Date"] == "16/08/2024"
    assert row["Time"] == "19:00"
    assert row["HomeTeam"] == "Man United"
    assert row["AwayTeam"] == "Fulham"
    assert row["FTHG"] == 1
    assert row["FTAG"] == 0
    assert row["FTR"] == "H"
    assert row["HTR"] == "D"
    assert row["Referee"] == "M. Oliver"


def test_transform_fixtures_derives_outcomes():
    """Tests that home, draw and away outcomes are all derived correctly."""
    def with_goals(home, away):
        return {
            "fixture": {"id": 1, "date": "2024-08-16T19:00:00+00:00"},
            "teams": {"home": {"name": "A"}, "away": {"name": "B"}},
            "goals": {"home": home, "away": away},
            "score": {"halftime": {"home": None, "away": None}},
        }

    response = {
        "response": [
            with_goals(2, 1),
            with_goals(1, 1),
            with_goals(0, 3),
        ]
    }

    result = api_football.transform_fixtures(response)

    assert result["FTR"].tolist() == ["H", "D", "A"]


def test_transform_fixtures_empty_response_has_columns():
    """
    Tests that an empty API response yields a correctly shaped empty frame.

    Building the frame from a list of rows produced a zero column DataFrame,
    which the CSV writer then rejected with a misleading complaint about a
    missing FixtureID column.
    """
    result = api_football.transform_fixtures({"response": []})

    assert result.empty
    assert "FixtureID" in result.columns
    assert "FTHG" in result.columns


def test_transform_fixtures_excludes_unplayed_by_default(unplayed_response):
    """
    Tests that a fixture with no final score is excluded.

    The feature builder and the Elo walk iterate every CSV row, so a scoreless
    row would be recorded as a 0-0 draw.
    """
    result = api_football.transform_fixtures({"response": [unplayed_response]})

    assert result.empty


def test_transform_fixtures_includes_unplayed_when_requested(unplayed_response):
    """Tests that unplayed fixtures can be written when explicitly asked for."""
    result = api_football.transform_fixtures(
        {"response": [unplayed_response]},
        include_unplayed=True,
    )

    assert len(result) == 1
    assert pd.isna(result.iloc[0]["FTHG"])


def test_transform_fixtures_mixes_played_and_unplayed(
    fixture_response,
    unplayed_response,
):
    """Tests that only the played fixture survives a mixed response."""
    response = {"response": [fixture_response, unplayed_response]}

    result = api_football.transform_fixtures(response)

    assert len(result) == 1
    assert result.iloc[0]["FTR"] == "H"


def test_transform_fixtures_maps_team_names(fixture_response):
    """
    Tests that API team names are translated to CSV spellings.

    The CSVs use football-data.co.uk short names. Without this a club appears
    twice in the data and its Elo rating and rolling form split in two.
    """
    fixture = dict(fixture_response)
    fixture["teams"] = {
        "home": {"name": "Manchester United"},
        "away": {"name": "Wolves"},
    }

    result = api_football.transform_fixtures(
        {"response": [fixture]},
        league="PremierLeague",
    )

    assert result.iloc[0]["HomeTeam"] == "Man United"
    assert result.iloc[0]["AwayTeam"] == "Wolves"


def test_transform_fixtures_uses_league_div_code(fixture_response):
    """Tests that the Div column uses the football-data.co.uk code."""
    result = api_football.transform_fixtures(
        {"response": [fixture_response]},
        league="PremierLeague",
    )

    assert result.iloc[0]["Div"] == "E0"


def test_transform_fixtures_writes_referee_for_premier_league(fixture_response):
    """Tests that the Referee column is produced for the Premier League."""
    result = api_football.transform_fixtures(
        {"response": [fixture_response]},
        league="PremierLeague",
    )

    assert "Referee" in result.columns
    assert result.iloc[0]["Referee"] == "M. Oliver"


def test_transform_fixtures_omits_referee_for_laliga(fixture_response):
    """
    Tests that the Referee column is omitted where the CSVs lack it.

    Writing it would add a column to the La Liga and Serie A files that
    nothing else in the project populates or reads.
    """
    result = api_football.transform_fixtures(
        {"response": [fixture_response]},
        league="LaLiga",
    )

    assert "Referee" not in result.columns


def test_transform_fixtures_with_statistics(fixture_response):
    """Tests that statistics are attached to the transformed row."""
    statistics = {
        1208021: {
            "home": {"Total Shots": 14, "Shots on Goal": 6},
            "away": {"Total Shots": 8, "Shots on Goal": 2},
        }
    }

    result = api_football.transform_fixtures(
        {"response": [fixture_response]},
        statistics,
    )

    row = result.iloc[0]

    assert row["HS"] == 14
    assert row["AS"] == 8
    assert row["HST"] == 6
    assert row["AST"] == 2


def test_transform_fixtures_missing_statistic_keys_do_not_crash(fixture_response):
    """Tests that a fixture missing some statistic keys still transforms."""
    statistics = {1208021: {"home": {"Total Shots": 14}, "away": {}}}

    result = api_football.transform_fixtures(
        {"response": [fixture_response]},
        statistics,
    )

    row = result.iloc[0]

    assert row["HS"] == 14
    assert row["AS"] is None
    assert row["HST"] is None


def test_transform_fixtures_excludes_xg(fixture_response):
    """
    Tests that transform_fixtures does not add xG columns.

    xG comes from Understat, and mixing two sources for the same feature would
    make the values inconsistent between seasons.
    """
    statistics = {
        1208021: {
            "home": {"Expected Goals": 1.8, "Total Shots": 14},
            "away": {"Expected Goals": 0.7, "Total Shots": 8},
        }
    }

    result = api_football.transform_fixtures(
        {"response": [fixture_response]},
        statistics,
    )

    assert "home_xg" not in result.columns
    assert "away_xg" not in result.columns


# ---------------------------------------------------------------------------
# update_csv_file
# ---------------------------------------------------------------------------


def test_update_csv_file_creates_csv(tmp_path, monkeypatch):
    """Tests that the CSV file is created when it does not yet exist."""
    monkeypatch.chdir(tmp_path)

    new_data = api_football.transform_fixtures(
        {
            "response": [
                {
                    "fixture": {
                        "id": 1,
                        "date": "2026-08-16T19:00:00+00:00",
                    },
                    "teams": {
                        "home": {"name": "Team A"},
                        "away": {"name": "Team B"},
                    },
                    "goals": {"home": 1, "away": 0},
                    "score": {"halftime": {"home": None, "away": None}},
                }
            ]
        },
        league="PremierLeague",
    )

    api_football.update_csv_file("PremierLeague", "2026-2027", new_data)

    csv_path = tmp_path / "football_data" / "PremierLeague" / "2026-2027.csv"

    assert csv_path.exists()
    assert len(pd.read_csv(csv_path)) == 1


def test_update_csv_file_creates_directory(tmp_path, monkeypatch):
    """Tests that the necessary directory structure is created."""
    monkeypatch.chdir(tmp_path)

    new_data = pd.DataFrame({
        "FixtureID": [1],
        "Date": ["16/08/2026"],
        "HomeTeam": ["Team A"],
        "AwayTeam": ["Team B"],
    })

    api_football.update_csv_file("LaLiga", "2026-2027", new_data)

    assert (tmp_path / "football_data" / "LaLiga" / "2026-2027.csv").exists()


def test_update_csv_file_appends_new_fixture(tmp_path, monkeypatch):
    """Tests that a fixture absent from the CSV is appended."""
    monkeypatch.chdir(tmp_path)

    csv_path = write_csv(
        tmp_path,
        "PremierLeague",
        "2026-2027",
        [
            {
                "Date": "16/08/2026",
                "HomeTeam": "Team A",
                "AwayTeam": "Team B",
                "FTHG": 1,
                "FTAG": 0,
                "FTR": "H",
            },
            {
                "Date": "16/08/2026",
                "HomeTeam": "Team B",
                "AwayTeam": "Team C",
                "FTHG": 0,
                "FTAG": 2,
                "FTR": "A",
            },
        ],
    )

    new_data = pd.DataFrame({
        "FixtureID": [2],
        "Date": ["17/08/2026"],
        "HomeTeam": ["Team A"],
        "AwayTeam": ["Team C"],
        "FTHG": [2],
        "FTAG": [1],
        "FTR": ["H"],
    })

    summary = api_football.update_csv_file(
        "PremierLeague", "2026-2027", new_data
    )

    result = pd.read_csv(csv_path)

    assert len(result) == 3
    assert summary["appended"] == 1
    assert "17/08/2026" in result["Date"].tolist()


def test_update_csv_file_never_overwrites_existing_values(tmp_path, monkeypatch):
    """
    Tests that existing values survive the merge.

    The football-data.co.uk files are the source of truth, and the API is only
    used to fill gaps. Overwriting would also let a corrected referee or a
    differing score definition rewrite settled history.
    """
    monkeypatch.chdir(tmp_path)

    csv_path = write_csv(
        tmp_path,
        "PremierLeague",
        "2026-2027",
        [
            {
                "Date": "16/08/2026",
                "HomeTeam": "Team A",
                "AwayTeam": "Team B",
                "FTHG": 1,
                "FTAG": 0,
                "FTR": "H",
                "Referee": "Original Referee",
            }
        ],
    )

    new_data = pd.DataFrame({
        "FixtureID": [1],
        "Date": ["16/08/2026"],
        "HomeTeam": ["Team A"],
        "AwayTeam": ["Team B"],
        "FTHG": [99],
        "FTAG": [99],
        "FTR": ["D"],
        "Referee": ["API Referee"],
    })

    api_football.update_csv_file("PremierLeague", "2026-2027", new_data)

    row = pd.read_csv(csv_path).iloc[0]

    assert row["FTHG"] == 1
    assert row["FTAG"] == 0
    assert row["FTR"] == "H"
    assert row["Referee"] == "Original Referee"


def test_update_csv_file_fills_empty_cells(tmp_path, monkeypatch):
    """Tests that empty cells in a matching row are filled from the API."""
    monkeypatch.chdir(tmp_path)

    csv_path = write_csv(
        tmp_path,
        "PremierLeague",
        "2026-2027",
        [
            {
                "Date": "16/08/2026",
                "HomeTeam": "Team A",
                "AwayTeam": "Team B",
                "FTHG": 1,
                "FTAG": 0,
                "FTR": "H",
                "Referee": None,
                "HS": None,
                "AS": None,
            }
        ],
    )

    new_data = pd.DataFrame({
        "FixtureID": [1],
        "Date": ["16/08/2026"],
        "HomeTeam": ["Team A"],
        "AwayTeam": ["Team B"],
        "Referee": ["M. Oliver"],
        "HS": [14],
        "AS": [8],
    })

    summary = api_football.update_csv_file(
        "PremierLeague", "2026-2027", new_data
    )

    row = pd.read_csv(csv_path).iloc[0]

    assert row["Referee"] == "M. Oliver"
    assert row["HS"] == 14
    assert row["AS"] == 8
    assert summary["filled"] == 3


def test_update_csv_file_matches_on_date_and_teams(tmp_path, monkeypatch):
    """
    Tests that rows are matched on date, home team and away team.

    The CSVs have no FixtureID column, so keying the merge on it left every
    existing row unmatched and appended a duplicate on each run.
    """
    monkeypatch.chdir(tmp_path)

    csv_path = write_csv(
        tmp_path,
        "PremierLeague",
        "2026-2027",
        [
            {
                "Date": "16/08/2026",
                "HomeTeam": "Team A",
                "AwayTeam": "Team B",
                "FTHG": 1,
                "FTAG": 0,
                "FTR": "H",
            }
        ],
    )

    new_data = pd.DataFrame({
        "FixtureID": [999],
        "Date": ["16/08/2026"],
        "HomeTeam": ["Team A"],
        "AwayTeam": ["Team B"],
        "Referee": ["M. Oliver"],
    })

    api_football.update_csv_file("PremierLeague", "2026-2027", new_data)

    result = pd.read_csv(csv_path)

    assert len(result) == 1
    assert result.iloc[0]["Referee"] == "M. Oliver"


def test_update_csv_file_is_idempotent(tmp_path, monkeypatch):
    """
    Tests that running the update twice does not duplicate rows.

    This is the regression test for the duplicate-row bug: with the old
    FixtureID key, every run appended the whole season again.
    """
    monkeypatch.chdir(tmp_path)

    csv_path = write_csv(
        tmp_path,
        "PremierLeague",
        "2026-2027",
        [
            {
                "Date": "16/08/2026",
                "HomeTeam": "Team A",
                "AwayTeam": "Team B",
                "FTHG": 1,
                "FTAG": 0,
                "FTR": "H",
                "Referee": None,
            }
        ],
    )

    new_data = pd.DataFrame({
        "FixtureID": [1],
        "Date": ["16/08/2026"],
        "HomeTeam": ["Team A"],
        "AwayTeam": ["Team B"],
        "Referee": ["M. Oliver"],
    })

    api_football.update_csv_file("PremierLeague", "2026-2027", new_data)
    first = csv_path.read_bytes()

    summary = api_football.update_csv_file(
        "PremierLeague", "2026-2027", new_data
    )

    assert len(pd.read_csv(csv_path)) == 1
    assert summary["appended"] == 0
    assert summary["filled"] == 0
    assert csv_path.read_bytes() == first


def test_update_csv_file_preserves_existing_columns(tmp_path, monkeypatch):
    """
    Tests that columns the API does not supply are preserved.

    The real CSVs carry over a hundred odds and handicap columns that a
    concat against the narrower API frame would have to keep intact.
    """
    monkeypatch.chdir(tmp_path)

    csv_path = write_csv(
        tmp_path,
        "PremierLeague",
        "2026-2027",
        [
            {
                "Date": "16/08/2026",
                "HomeTeam": "Team A",
                "AwayTeam": "Team B",
                "FTHG": 1,
                "FTAG": 0,
                "FTR": "H",
                "B365H": 1.8,
                "AvgA": 4.2,
            },
            {
                "Date": "17/08/2026",
                "HomeTeam": "Team A",
                "AwayTeam": "Team C",
                "FTHG": 0,
                "FTAG": 0,
                "FTR": "D",
                "B365H": 1.9,
                "AvgA": 4.5,
            },
        ],
    )

    new_data = pd.DataFrame({
        "FixtureID": [1],
        "Date": ["16/08/2026"],
        "HomeTeam": ["Team A"],
        "AwayTeam": ["Team B"],
        "Referee": ["M. Oliver"],
    })

    api_football.update_csv_file("PremierLeague", "2026-2027", new_data)

    result = pd.read_csv(csv_path)

    assert "B365H" in result.columns
    assert "AvgA" in result.columns
    assert result["B365H"].tolist() == [1.8, 1.9]
    assert result["AvgA"].tolist() == [4.2, 4.5]


def test_update_csv_file_does_not_append_unplayed_rows(tmp_path, monkeypatch):
    """
    Tests that a scoreless API fixture is not written to the CSV.
    """
    monkeypatch.chdir(tmp_path)

    csv_path = write_csv(
        tmp_path,
        "PremierLeague",
        "2026-2027",
        [
            {
                "Date": "16/08/2026",
                "HomeTeam": "Team A",
                "AwayTeam": "Team B",
                "FTHG": 1,
                "FTAG": 0,
                "FTR": "H",
            },
            {
                "Date": "16/08/2026",
                "HomeTeam": "Team B",
                "AwayTeam": "Team C",
                "FTHG": 0,
                "FTAG": 2,
                "FTR": "A",
            },
        ],
    )

    new_data = pd.DataFrame({
        "FixtureID": [2],
        "Date": ["20/12/2026"],
        "HomeTeam": ["Team A"],
        "AwayTeam": ["Team C"],
        "FTHG": [None],
        "FTAG": [None],
        "FTR": [None],
    })

    api_football.update_csv_file("PremierLeague", "2026-2027", new_data)

    assert len(pd.read_csv(csv_path)) == 2
    assert "20/12/2026" not in pd.read_csv(csv_path)["Date"].tolist()


def test_update_csv_file_rejects_unknown_teams(tmp_path, monkeypatch):
    """
    Tests that a team the CSVs have never seen is rejected.

    Ingesting it would create a second entry for a club that already exists
    under another name, splitting its Elo rating and rolling form.
    """
    monkeypatch.chdir(tmp_path)

    write_csv(
        tmp_path,
        "PremierLeague",
        "2026-2027",
        [
            {
                "Date": "16/08/2026",
                "HomeTeam": "Team A",
                "AwayTeam": "Team B",
                "FTHG": 1,
                "FTAG": 0,
                "FTR": "H",
            }
        ],
    )

    new_data = pd.DataFrame({
        "FixtureID": [2],
        "Date": ["17/08/2026"],
        "HomeTeam": ["Wigan Athletic"],
        "AwayTeam": ["Team B"],
        "FTHG": [1],
        "FTAG": [0],
        "FTR": ["H"],
    })

    with pytest.raises(api_football.APIFootballError, match="Wigan Athletic"):
        api_football.update_csv_file("PremierLeague", "2026-2027", new_data)


def test_update_csv_file_empty_data_is_a_no_op(tmp_path, monkeypatch):
    """
    Tests that empty input neither raises nor writes.

    A date window with no completed matches is routine, and it used to fail
    with a complaint about a missing FixtureID column.
    """
    monkeypatch.chdir(tmp_path)

    csv_path = write_csv(
        tmp_path,
        "PremierLeague",
        "2026-2027",
        [
            {
                "Date": "16/08/2026",
                "HomeTeam": "Team A",
                "AwayTeam": "Team B",
            }
        ],
    )

    before = csv_path.read_bytes()

    summary = api_football.update_csv_file(
        "PremierLeague",
        "2026-2027",
        api_football.transform_fixtures({"response": []}),
    )

    assert summary == {"filled": 0, "appended": 0, "unchanged": 0}
    assert csv_path.read_bytes() == before


def test_update_csv_file_dry_run_does_not_write(tmp_path, monkeypatch):
    """Tests that a dry run reports the change without writing the file."""
    monkeypatch.chdir(tmp_path)

    csv_path = write_csv(
        tmp_path,
        "PremierLeague",
        "2026-2027",
        [
            {
                "Date": "16/08/2026",
                "HomeTeam": "Team A",
                "AwayTeam": "Team B",
                "FTHG": 1,
                "FTAG": 0,
                "FTR": "H",
                "Referee": None,
            }
        ],
    )

    before = csv_path.read_bytes()

    new_data = pd.DataFrame({
        "FixtureID": [1],
        "Date": ["16/08/2026"],
        "HomeTeam": ["Team A"],
        "AwayTeam": ["Team B"],
        "Referee": ["M. Oliver"],
    })

    summary = api_football.update_csv_file(
        "PremierLeague", "2026-2027", new_data, dry_run=True
    )

    assert summary["filled"] == 1
    assert csv_path.read_bytes() == before


def test_update_csv_file_rejects_missing_fixture_id_column(tmp_path, monkeypatch):
    """Tests that input without a FixtureID column is rejected."""
    monkeypatch.chdir(tmp_path)

    new_data = pd.DataFrame({
        "Date": ["16/08/2026"],
        "HomeTeam": ["Team A"],
        "AwayTeam": ["Team B"],
    })

    with pytest.raises(ValueError, match="FixtureID"):
        api_football.update_csv_file("PremierLeague", "2026-2027", new_data)


def test_update_csv_file_rejects_null_fixture_id(tmp_path, monkeypatch):
    """Tests that input with a missing FixtureID is rejected."""
    monkeypatch.chdir(tmp_path)

    new_data = pd.DataFrame({
        "FixtureID": [None],
        "Date": ["16/08/2026"],
        "HomeTeam": ["Team A"],
        "AwayTeam": ["Team B"],
    })

    with pytest.raises(ValueError, match="without a FixtureID"):
        api_football.update_csv_file("PremierLeague", "2026-2027", new_data)


# ---------------------------------------------------------------------------
# update_league_data
# ---------------------------------------------------------------------------


def test_update_league_data_writes_csv(
    tmp_path,
    monkeypatch,
    fixture_response,
):
    """Tests that update_league_data fetches and stores league data."""
    schedule_params = []

    def mock_get(url, params=None, headers=None, timeout=None):
        if params and "ids" in params:
            fixture_ids = params["ids"].split("-")

            return MockResponse({
                "response": [
                    {
                        "fixture": {"id": int(fixture_id)},
                        "teams": {"home": {"id": 33}, "away": {"id": 34}},
                        "statistics": [
                            {
                                "team": {"id": 33},
                                "statistics": [
                                    {"type": "Total Shots", "value": 14}
                                ],
                            },
                            {
                                "team": {"id": 34},
                                "statistics": [
                                    {"type": "Total Shots", "value": 8}
                                ],
                            },
                        ],
                    }
                    for fixture_id in fixture_ids
                ]
            })

        schedule_params.append(params)

        return MockResponse({"response": [fixture_response]})

    mock_session_get(monkeypatch, mock_get)
    monkeypatch.chdir(tmp_path)

    api_football.update_league_data(
        39, "2024-2025", from_date="2024-08-16", to_date="2024-08-16"
    )

    assert len(schedule_params) == 1
    assert schedule_params[0]["status"] == api_football.COMPLETED_STATUSES
    assert schedule_params[0]["from"] == "2024-08-16"

    csv_path = tmp_path / "football_data" / "PremierLeague" / "2024-2025.csv"

    assert csv_path.exists()

    row = pd.read_csv(csv_path).iloc[0]

    assert row["FixtureID"] == 1208021
    assert row["FTHG"] == 1
    assert row["HS"] == 14


def test_update_league_data_survives_missing_statistics(
    tmp_path,
    monkeypatch,
    fixture_response,
):
    """
    Tests that a rejected statistics call still ingests the results.

    Some plans do not allow the ids parameter, and the fixtures endpoint does
    not return statistics inline, so the scores must not be lost with them.
    """
    def mock_get(url, params=None, headers=None, timeout=None):
        if params and "ids" in params:
            return MockResponse(
                {"errors": {"plan": "Free plans do not have access to the Ids"}}
            )

        return MockResponse({"response": [fixture_response]})

    mock_session_get(monkeypatch, mock_get)
    monkeypatch.chdir(tmp_path)

    api_football.update_league_data(39, 2024, "PremierLeague")

    csv_path = tmp_path / "football_data" / "PremierLeague" / "2024-2025.csv"

    row = pd.read_csv(csv_path).iloc[0]

    assert row["FTHG"] == 1
    assert pd.isna(row["HS"])


def test_update_league_data_no_fixtures_writes_nothing(
    tmp_path,
    monkeypatch,
):
    """
    Tests that a window with no completed matches is a clean no-op.

    Mondays, pre-season and postponed matchdays all produce an empty
    response, which previously crashed the CSV writer.
    """
    def mock_get(url, params=None, headers=None, timeout=None):
        return MockResponse({"response": []})

    mock_session_get(monkeypatch, mock_get)
    monkeypatch.chdir(tmp_path)

    summary = api_football.update_league_data(39, 2024, "PremierLeague")

    assert summary == {"filled": 0, "appended": 0, "unchanged": 0}
    assert not (tmp_path / "football_data").exists()


def test_update_league_data_handles_fixture_without_id(tmp_path, monkeypatch):
    """
    Tests that a fixture missing its nested id is skipped rather than raising.
    """
    def mock_get(url, params=None, headers=None, timeout=None):
        if params and "ids" in params:
            return MockResponse({"response": []})

        return MockResponse({
            "response": [
                {
                    "fixture": None,
                    "teams": {"home": {"name": "A"}, "away": {"name": "B"}},
                    "goals": {"home": 1, "away": 0},
                },
                {
                    "fixture": {"id": 5, "date": "2024-08-16T19:00:00+00:00"},
                    "teams": {"home": {"name": "A"}, "away": {"name": "B"}},
                    "goals": {"home": 1, "away": 0},
                    "score": {"halftime": {"home": None, "away": None}},
                },
            ]
        })

    mock_session_get(monkeypatch, mock_get)
    monkeypatch.chdir(tmp_path)

    api_football.update_league_data(39, 2024, "PremierLeague")

    csv_path = tmp_path / "football_data" / "PremierLeague" / "2024-2025.csv"

    assert len(pd.read_csv(csv_path)) == 1


def test_update_league_data_unknown_league_id_raises(monkeypatch):
    """Tests that an unmapped league ID raises a clear error."""
    with pytest.raises(api_football.APIFootballError, match="Unknown league_id"):
        api_football.update_league_data(999, 2024)


# ---------------------------------------------------------------------------
# create_initial_season_data
# ---------------------------------------------------------------------------


def test_create_initial_season_data_writes_csv(
    tmp_path,
    monkeypatch,
    fixture_response,
    unplayed_response,
):
    """
    Tests that create_initial_season_data writes the played fixtures.

    The unplayed fixture in the same response must be left out.
    """
    detail_ids = []

    def mock_get(url, params=None, headers=None, timeout=None):
        if params and "ids" in params:
            detail_ids.extend(params["ids"].split("-"))

            return MockResponse({"response": []})

        return MockResponse({
            "response": [fixture_response, unplayed_response]
        })

    mock_session_get(monkeypatch, mock_get)
    monkeypatch.chdir(tmp_path)

    api_football.create_initial_season_data(39, "2024-2025")

    csv_path = tmp_path / "football_data" / "PremierLeague" / "2024-2025.csv"

    result = pd.read_csv(csv_path)

    assert len(result) == 1
    assert result.iloc[0]["FixtureID"] == 1208021
    assert detail_ids == []


def test_create_initial_season_data_includes_unplayed_when_requested(
    tmp_path,
    monkeypatch,
    unplayed_response,
):
    """Tests that the whole schedule can be written when explicitly asked."""
    def mock_get(url, params=None, headers=None, timeout=None):
        return MockResponse({"response": [unplayed_response]})

    mock_session_get(monkeypatch, mock_get)
    monkeypatch.chdir(tmp_path)

    api_football.create_initial_season_data(
        39, "2024-2025", include_unplayed=True
    )

    csv_path = tmp_path / "football_data" / "PremierLeague" / "2024-2025.csv"

    assert len(pd.read_csv(csv_path)) == 1


def test_create_initial_season_data_empty_schedule(tmp_path, monkeypatch):
    """Tests that an empty schedule is a clean no-op."""

    def mock_get(url, params=None, headers=None, timeout=None):
        return MockResponse({"response": []})

    mock_session_get(monkeypatch, mock_get)
    monkeypatch.chdir(tmp_path)

    summary = api_football.create_initial_season_data(39, 2024)

    assert summary == {"filled": 0, "appended": 0, "unchanged": 0}


def test_create_initial_season_data_unknown_league_id_raises(monkeypatch):
    """Tests that an unmapped league ID raises a clear error."""
    with pytest.raises(api_football.APIFootballError, match="Unknown league_id"):
        api_football.create_initial_season_data(999, 2024)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_league_ids_cover_every_configured_league():
    """Tests that every configured league has an API-Football ID."""
    from config import LEAGUES

    assert set(api_football.LEAGUE_IDS) == set(LEAGUES)


def test_only_premier_league_writes_referee():
    """
    Tests that Referee is limited to the Premier League.

    Only the Premier League CSVs have the column, and the value is read by
    nothing in the pipeline, so writing it elsewhere would only change the
    file schema.
    """
    from config import LEAGUES

    for league, config in LEAGUES.items():
        has_column = "Referee" in pd.read_csv(
            config["football_data"] + "/2026-2027.csv", nrows=0
        ).columns

        assert has_column == config["writes_referee"], league
