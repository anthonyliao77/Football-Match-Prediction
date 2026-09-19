"""
Tests for the API-Football module.
"""

import pandas as pd
import pytest

from src import api_football


class MockResponse:
    """Minimal stand-in for a requests.Response object."""

    def __init__(self, data, status_code=200):
        self.data = data
        self.status_code = status_code

    def json(self):
        return self.data


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


def test_update_csv_file_creates_csv(tmp_path, monkeypatch):
    """
    Test that the update_csv_file function creates a CSV file
    with the correct data.
    """

    monkeypatch.chdir(tmp_path)

    new_data = pd.DataFrame({
        "FixtureID": [1, 2],
        "HomeTeam": ["Team A", "Team C"],
        "AwayTeam": ["Team B", "Team D"],
    })

    api_football.update_csv_file("PremierLeague", "2026-2027", new_data)

    csv_path = tmp_path / "football_data" / "PremierLeague" / "2026-2027.csv"

    assert csv_path.exists()

    result = pd.read_csv(csv_path)

    pd.testing.assert_frame_equal(result, new_data)


def test_update_csv_file_creates_directory(tmp_path, monkeypatch):
    """
    Test that the update_csv_file function creates the
    necessary directory structure.
    """

    monkeypatch.chdir(tmp_path)

    new_data = pd.DataFrame({
        "FixtureID": [1],
        "HomeTeam": ["Team A"],
        "AwayTeam": ["Team B"],
    })

    api_football.update_csv_file("LaLiga", "2026-2027", new_data)

    csv_path = tmp_path / "football_data" / "LaLiga" / "2026-2027.csv"

    assert csv_path.exists()


def test_update_csv_file_appends_new_fixtures(tmp_path, monkeypatch):
    """
    Test that the update_csv_file function appends new fixtures.
    """

    monkeypatch.chdir(tmp_path)

    csv_path = tmp_path / "football_data" / "PremierLeague" / "2026-2027.csv"
    csv_path.parent.mkdir(parents=True)

    existing_data = pd.DataFrame({
        "FixtureID": [1],
        "HomeTeam": ["Team A"],
        "AwayTeam": ["Team B"],
    })

    new_data = pd.DataFrame({
        "FixtureID": [2],
        "HomeTeam": ["Team C"],
        "AwayTeam": ["Team D"],
    })

    existing_data.to_csv(csv_path, index=False)

    api_football.update_csv_file("PremierLeague", "2026-2027", new_data)
    result = pd.read_csv(csv_path)

    assert len(result) == 2
    assert result["FixtureID"].tolist() == [1, 2]


def test_update_csv_file_updates_existing_fixture(tmp_path, monkeypatch):
    """
    Test that the update_csv_file function updates existing fixtures.
    """

    monkeypatch.chdir(tmp_path)

    csv_path = tmp_path / "football_data" / "PremierLeague" / "2026-2027.csv"
    csv_path.parent.mkdir(parents=True)

    existing_data = pd.DataFrame({
        "FixtureID": [1],
        "HomeTeam": ["Team A"],
        "AwayTeam": ["Team B"],
        "FTHG": [None],
        "FTAG": [None],
    })

    new_data = pd.DataFrame({
        "FixtureID": [1],
        "HomeTeam": ["Team A"],
        "AwayTeam": ["Team B"],
        "FTHG": [2],
        "FTAG": [1],
    })

    existing_data.to_csv(csv_path, index=False)

    api_football.update_csv_file("PremierLeague", "2026-2027", new_data)

    result = pd.read_csv(csv_path)

    assert len(result) == 1
    assert result.loc[0, "FixtureID"] == 1
    assert result.loc[0, "FTHG"] == 2
    assert result.loc[0, "FTAG"] == 1


def test_update_csv_file_does_not_duplicate_fixture(tmp_path, monkeypatch):
    """
    Test that the update_csv_file function does not duplicate fixtures.
    """

    monkeypatch.chdir(tmp_path)

    csv_path = tmp_path / "football_data" / "SerieA" / "2026-2027.csv"
    csv_path.parent.mkdir(parents=True)

    existing_data = pd.DataFrame({
        "FixtureID": [1, 2],
        "HomeTeam": ["Team A", "Team C"],
        "AwayTeam": ["Team B", "Team D"],
    })

    new_data = pd.DataFrame({
        "FixtureID": [2],
        "HomeTeam": ["Team C"],
        "AwayTeam": ["Team D"],
    })

    existing_data.to_csv(csv_path, index=False)
    api_football.update_csv_file("SerieA", "2026-2027", new_data)

    result = pd.read_csv(csv_path)

    assert len(result) == 2
    assert result["FixtureID"].tolist() == [1, 2]


def test_fetch_league_schedule_requests_fixtures(monkeypatch, fixture_response):
    """Tests that fetch_league_schedule fetches the full season schedule."""

    captured = {}

    def mock_get(url, params=None, headers=None, timeout=None):
        captured["url"] = url
        captured["params"] = params

        return MockResponse({
            "results": 1,
            "paging": {"current": 1, "total": 1},
            "response": [fixture_response],
        })

    monkeypatch.setattr(api_football.requests, "get", mock_get)

    result = api_football.fetch_league_schedule(39, "2024-2025")

    assert captured["url"] == f"{api_football.API_BASE_URL}/fixtures"
    assert captured["params"]["league"] == 39
    assert captured["params"]["season"] == 2024
    assert "status" not in captured["params"]
    assert result["response"] == [fixture_response]


def test_fetch_league_schedule_passes_date_range(monkeypatch, fixture_response):
    """Tests that fetch_league_schedule forwards the date range."""

    captured = {}

    def mock_get(url, params=None, headers=None, timeout=None):
        captured["params"] = params

        return MockResponse({
            "paging": {"current": 1, "total": 1},
            "response": [fixture_response],
        })

    monkeypatch.setattr(api_football.requests, "get", mock_get)

    api_football.fetch_league_schedule(
        39, 2024, from_date="2024-09-01", to_date="2024-09-30"
    )

    assert captured["params"]["from"] == "2024-09-01"
    assert captured["params"]["to"] == "2024-09-30"


def test_fetch_completed_fixtures_requests_fixtures(monkeypatch, fixture_response):
    """Tests that fetch_completed_fixtures filters by completed status."""

    captured = {}

    def mock_get(url, params=None, headers=None, timeout=None):
        captured["params"] = params

        return MockResponse({
            "paging": {"current": 1, "total": 1},
            "response": [fixture_response],
        })

    monkeypatch.setattr(api_football.requests, "get", mock_get)

    result = api_football.fetch_completed_fixtures(
        39, 2024, from_date="2024-09-01", to_date="2024-09-15"
    )

    assert captured["params"]["status"] == api_football.COMPLETED_STATUSES
    assert captured["params"]["from"] == "2024-09-01"
    assert captured["params"]["to"] == "2024-09-15"
    assert result["response"] == [fixture_response]


def test_fetch_league_schedule_does_not_send_page(monkeypatch, fixture_response):
    """Tests that fetch_league_schedule does not send a page parameter."""

    captured = {}

    def mock_get(url, params=None, headers=None, timeout=None):
        captured["params"] = params

        return MockResponse({
            "paging": {"current": 1, "total": 1},
            "response": [fixture_response],
        })

    monkeypatch.setattr(api_football.requests, "get", mock_get)

    result = api_football.fetch_league_schedule(39, 2024)

    assert "page" not in captured["params"]
    assert result["response"] == [fixture_response]


def test_fetch_completed_fixtures_does_not_send_page(
    monkeypatch,
    fixture_response,
):
    """Tests that fetch_completed_fixtures does not send a page parameter."""

    captured = {}

    def mock_get(url, params=None, headers=None, timeout=None):
        captured["params"] = params

        return MockResponse({
            "paging": {"current": 1, "total": 1},
            "response": [fixture_response],
        })

    monkeypatch.setattr(api_football.requests, "get", mock_get)

    result = api_football.fetch_completed_fixtures(39, 2024)

    assert "page" not in captured["params"]
    assert result["response"] == [fixture_response]


def test_fetch_league_schedule_raises_on_api_error(monkeypatch):
    """Tests that API errors raise an APIFootballError."""

    def mock_get(url, params=None, headers=None, timeout=None):
        return MockResponse({
            "errors": {"request": "Invalid league"},
        })

    monkeypatch.setattr(api_football.requests, "get", mock_get)

    with pytest.raises(api_football.APIFootballError):
        api_football.fetch_league_schedule(39, 2024)


def test_fetch_fixture_details_batches_statistics(monkeypatch):
    """
    Tests that fetch_fixture_details batches IDs and parses the
    statistics returned by the enriched fixtures response.
    """

    requested_ids = []

    def mock_get(url, params=None, headers=None, timeout=None):
        requested_ids.append(params["ids"])

        return MockResponse({
            "response": [
                {
                    "fixture": {"id": 1208021},
                    "teams": {
                        "home": {"id": 33, "name": "Man United"},
                        "away": {"id": 34, "name": "Fulham"},
                    },
                    "statistics": [
                        {
                            "team": {"id": 33, "name": "Man United"},
                            "statistics": [
                                {"type": "Total Shots", "value": 14},
                                {"type": "Shots on Goal", "value": 5},
                                {"type": "Fouls", "value": 8},
                            ],
                        },
                        {
                            "team": {"id": 34, "name": "Fulham"},
                            "statistics": [
                                {"type": "Total Shots", "value": 8},
                                {"type": "Shots on Goal", "value": 2},
                            ],
                        },
                    ],
                },
                {
                    "fixture": {"id": 1208022},
                    "teams": {
                        "home": {"id": 33, "name": "Man United"},
                        "away": {"id": 34, "name": "Fulham"},
                    },
                    "statistics": [],
                },
            ]
        })

    monkeypatch.setattr(api_football.requests, "get", mock_get)

    result = api_football.fetch_fixture_details([1208021, 1208022])

    assert requested_ids == ["1208021-1208022"]
    assert result[1208021]["home"]["Total Shots"] == 14
    assert result[1208021]["away"]["Shots on Goal"] == 2
    assert result[1208022] == {"home": {}, "away": {}}


def test_fetch_fixture_details_chunks_ids(monkeypatch):
    """
    Tests that fetch_fixture_details splits IDs into chunks of at most
    MAX_IDS_PER_REQUEST and makes one request per chunk.
    """

    call_count = 0

    def mock_get(url, params=None, headers=None, timeout=None):
        nonlocal call_count
        call_count += 1

        chunk_ids = params["ids"].split("-")

        assert len(chunk_ids) <= api_football.MAX_IDS_PER_REQUEST

        return MockResponse({
            "response": [
                {
                    "fixture": {"id": int(fixture_id)},
                    "teams": {"home": {"id": 33}, "away": {"id": 34}},
                    "statistics": [],
                }
                for fixture_id in chunk_ids
            ]
        })

    monkeypatch.setattr(api_football.requests, "get", mock_get)

    fixture_ids = list(range(1, 46))

    result = api_football.fetch_fixture_details(fixture_ids)

    assert len(result) == 45
    assert call_count == 3


def test_fetch_fixture_details_assigns_sides_by_team_id(monkeypatch):
    """
    Tests that home/away statistics are assigned by team ID rather than
    relying on the order of the statistics blocks.
    """

    def mock_get(url, params=None, headers=None, timeout=None):
        return MockResponse({
            "response": [
                {
                    "fixture": {"id": 1208021},
                    "teams": {
                        "home": {"id": 33, "name": "Man United"},
                        "away": {"id": 34, "name": "Fulham"},
                    },
                    "statistics": [
                        {
                            "team": {"id": 34, "name": "Fulham"},
                            "statistics": [
                                {"type": "Total Shots", "value": 8}
                            ],
                        },
                        {
                            "team": {"id": 33, "name": "Man United"},
                            "statistics": [
                                {"type": "Total Shots", "value": 14}
                            ],
                        },
                    ],
                }
            ]
        })

    monkeypatch.setattr(api_football.requests, "get", mock_get)

    result = api_football.fetch_fixture_details([1208021])

    assert result[1208021]["home"]["Total Shots"] == 14
    assert result[1208021]["away"]["Total Shots"] == 8


def test_transform_fixtures(fixture_response):
    """Tests that transform_fixtures flattens the response into a DataFrame."""

    response = {"response": [fixture_response]}

    result = api_football.transform_fixtures(response)

    assert len(result) == 1

    row = result.iloc[0]

    assert row["FixtureID"] == 1208021
    assert row["Date"] == "16/08/2024"
    assert row["Time"] == "19:00"
    assert row["HomeTeam"] == "Man United"
    assert row["AwayTeam"] == "Fulham"
    assert row["FTHG"] == 1
    assert row["FTAG"] == 0
    assert row["FTR"] == "H"
    assert row["HTHG"] == 0
    assert row["HTAG"] == 0
    assert row["HTR"] == "D"


def test_transform_fixtures_with_statistics(fixture_response):
    """Tests that transform_fixtures merges fixture statistics when given."""

    response = {"response": [fixture_response]}

    statistics = {
        1208021: {
            "home": {"Total Shots": 14, "Shots on Goal": 5},
            "away": {"Total Shots": 8, "Shots on Goal": 2},
        }
    }

    result = api_football.transform_fixtures(response, statistics)

    row = result.iloc[0]

    assert row["HS"] == 14
    assert row["AS"] == 8
    assert row["HST"] == 5
    assert row["AST"] == 2


def test_transform_fixtures_draw_and_away_outcomes():
    """Tests that draw and away-win outcomes are derived correctly."""

    fixtures = [
        {
            "fixture": {"id": 1, "date": "2024-08-17T12:30:00+00:00"},
            "league": {"id": 39, "name": "Premier League"},
            "teams": {
                "home": {"name": "Ipswich"},
                "away": {"name": "Liverpool"},
            },
            "goals": {"home": 0, "away": 2},
            "score": {"halftime": {"home": 0, "away": 0}},
        },
        {
            "fixture": {"id": 2, "date": "2024-08-18T14:00:00+00:00"},
            "league": {"id": 39, "name": "Premier League"},
            "teams": {
                "home": {"name": "Chelsea"},
                "away": {"name": "Arsenal"},
            },
            "goals": {"home": 1, "away": 1},
            "score": {"halftime": {"home": 1, "away": 1}},
        },
    ]

    result = api_football.transform_fixtures({"response": fixtures})

    assert result.iloc[0]["FTR"] == "A"
    assert result.iloc[1]["FTR"] == "D"


def test_transform_fixtures_missing_statistic_keys_do_not_crash(fixture_response):
    """
    Tests that a fixture missing some statistic keys still transforms cleanly.
    """

    response = {"response": [fixture_response]}

    statistics = {1208021: {"home": {"Total Shots": 14}, "away": {}}}

    result = api_football.transform_fixtures(response, statistics)

    row = result.iloc[0]

    assert row["HS"] == 14
    assert row["AS"] is None
    assert row["HST"] is None
    assert row["AST"] is None


def test_transform_fixtures_excludes_xg(fixture_response):
    """
    Tests that transform_fixtures does not add unverified xG columns.
    """

    response = {"response": [fixture_response]}

    statistics = {
        1208021: {
            "home": {"Expected Goals": 1.8, "Total Shots": 14},
            "away": {"Expected Goals": 0.7, "Total Shots": 8},
        }
    }

    result = api_football.transform_fixtures(response, statistics)

    assert "home_xg" not in result.columns
    assert "away_xg" not in result.columns
    assert result.iloc[0]["HS"] == 14


def test_update_league_data_writes_csv(
    tmp_path,
    monkeypatch,
    fixture_response,
):
    """Tests that update_league_data fetches and stores league data."""

    schedule_params = []

    def mock_get(url, params=None, headers=None, timeout=None):
        if "ids" in params:
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

        return MockResponse({
            "paging": {"current": 1, "total": 1},
            "response": [fixture_response],
        })

    monkeypatch.setattr(api_football.requests, "get", mock_get)
    monkeypatch.chdir(tmp_path)

    api_football.update_league_data(
        39, "2024-2025", from_date="2024-08-16", to_date="2024-08-16"
    )

    assert len(schedule_params) == 1
    assert schedule_params[0]["status"] == api_football.COMPLETED_STATUSES
    assert schedule_params[0]["from"] == "2024-08-16"
    assert schedule_params[0]["to"] == "2024-08-16"

    csv_path = tmp_path / "football_data" / "PremierLeague" / "2024-2025.csv"

    assert csv_path.exists()

    result = pd.read_csv(csv_path)

    assert len(result) == 1
    assert result.loc[0, "FixtureID"] == 1208021
    assert result.loc[0, "FTHG"] == 1


def test_create_initial_season_data_writes_csv(
    tmp_path,
    monkeypatch,
    fixture_response,
):
    """Tests that create_initial_season_data writes the full season schedule."""

    schedule_params = []
    detail_ids = []

    def mock_get(url, params=None, headers=None, timeout=None):
        if "ids" in params:
            detail_ids.append(params["ids"])

            return MockResponse({"response": []})

        schedule_params.append(params)

        return MockResponse({
            "paging": {"current": 1, "total": 1},
            "response": [fixture_response],
        })

    monkeypatch.setattr(api_football.requests, "get", mock_get)
    monkeypatch.chdir(tmp_path)

    api_football.create_initial_season_data(39, "2024-2025")

    assert detail_ids == []
    assert "status" not in schedule_params[0]

    csv_path = tmp_path / "football_data" / "PremierLeague" / "2024-2025.csv"

    assert csv_path.exists()


def test_update_league_data_unknown_league_id_raises(
    tmp_path,
    monkeypatch,
    fixture_response,
):
    """Tests that an unknown league_id raises instead of inferring the name."""

    def mock_get(url, params=None, headers=None, timeout=None):
        if "ids" in params:
            return MockResponse({"response": []})

        return MockResponse({
            "paging": {"current": 1, "total": 1},
            "response": [fixture_response],
        })

    monkeypatch.setattr(api_football.requests, "get", mock_get)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(api_football.APIFootballError, match="league_id"):
        api_football.update_league_data(999, "2024-2025")


def test_create_initial_season_data_unknown_league_id_raises(
    tmp_path,
    monkeypatch,
):
    """Tests that create_initial_season_data rejects an unknown league_id."""

    def mock_get(url, params=None, headers=None, timeout=None):
        return MockResponse({
            "paging": {"current": 1, "total": 1},
            "response": [],
        })

    monkeypatch.setattr(api_football.requests, "get", mock_get)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(api_football.APIFootballError, match="league_id"):
        api_football.create_initial_season_data(999, "2024-2025")


def test_csv_season_formats():
    """Tests the _csv_season helper across supported season formats."""

    assert api_football._csv_season(2026) == "2026-2027"
    assert api_football._csv_season("2026") == "2026-2027"
    assert api_football._csv_season("2026-2027") == "2026-2027"
    assert api_football._csv_season("2026/2027") == "2026-2027"


def test_update_csv_file_uses_csv_season(tmp_path, monkeypatch):
    """Tests that an integer season maps to the YYYY-YYYY CSV filename."""

    monkeypatch.chdir(tmp_path)

    new_data = pd.DataFrame({
        "FixtureID": [1],
        "HomeTeam": ["Team A"],
        "AwayTeam": ["Team B"],
    })

    api_football.update_csv_file("PremierLeague", 2026, new_data)

    csv_path = tmp_path / "football_data" / "PremierLeague" / "2026-2027.csv"

    assert csv_path.exists()


def test_missing_api_key_raises(monkeypatch):
    """Tests that a missing API key raises a clear APIFootballError."""

    monkeypatch.setattr(
        api_football.os, "getenv", lambda name, default=None: None
    )

    with pytest.raises(api_football.APIFootballError, match="API_FOOTBALL_KEY"):
        api_football._require_api_key()


def test_update_csv_file_rejects_missing_fixture_id(tmp_path, monkeypatch):
    """Tests that rows without a FixtureID are rejected."""

    monkeypatch.chdir(tmp_path)

    new_data = pd.DataFrame({
        "FixtureID": [1, None],
        "HomeTeam": ["Team A", "Team C"],
        "AwayTeam": ["Team B", "Team D"],
    })

    with pytest.raises(ValueError, match="FixtureID"):
        api_football.update_csv_file("PremierLeague", "2026-2027", new_data)


def test_update_csv_file_rejects_missing_fixture_id_column(tmp_path, monkeypatch):
    """Tests that data without a FixtureID column is rejected."""

    monkeypatch.chdir(tmp_path)

    new_data = pd.DataFrame({
        "HomeTeam": ["Team A"],
        "AwayTeam": ["Team B"],
    })

    with pytest.raises(ValueError, match="FixtureID column"):
        api_football.update_csv_file("PremierLeague", "2026-2027", new_data)
