"""
Tests for the API-Football module.
"""

import pandas as pd

from src import api_football


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
