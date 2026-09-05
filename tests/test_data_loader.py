"""
Tests for the football data loader.
"""

import pandas as pd

# pyrefly: ignore [missing-import]
import pytest

from src import data_loader


def test_load_data_combines_csv_files(tmp_path, monkeypatch):
    """Test that multiple CSV files are loaded and combined."""

    league_path = tmp_path / "PremierLeague"
    league_path.mkdir()

    csv_1 = league_path / "2022.csv"
    csv_2 = league_path / "2023.csv"

    csv_1.write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
        "01/08/2022,Team A,Team B,2,1,H\n"
    )

    csv_2.write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
        "02/08/2023,Team C,Team D,1,1,D\n"
    )

    monkeypatch.chdir(tmp_path)

    result = data_loader.load_data("PremierLeague")

    assert len(result) == 2
    assert list(result["HomeTeam"]) == ["Team A", "Team C"]
    assert list(result["AwayTeam"]) == ["Team B", "Team D"]


def test_load_data_normalizes_dates(tmp_path, monkeypatch):
    """Test that dates are converted to normalized pandas timestamps."""

    league_path = tmp_path / "PremierLeague"
    league_path.mkdir()

    csv_file = league_path / "2023.csv"

    csv_file.write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
        "15/08/2023,Team A,Team B,2,1,H\n"
    )

    monkeypatch.chdir(tmp_path)

    result = data_loader.load_data("PremierLeague")

    assert result.loc[0, "Date"] == pd.Timestamp("2023-08-15")
    assert result.loc[0, "Date"].hour == 0
    assert result.loc[0, "Date"].minute == 0
    assert result.loc[0, "Date"].second == 0


def test_load_data_converts_team_names_to_strings(tmp_path, monkeypatch):
    """Test that home and away team names are strings."""

    league_path = tmp_path / "PremierLeague"
    league_path.mkdir()

    csv_file = league_path / "2023.csv"

    csv_file.write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
        "15/08/2023,123,456,2,1,H\n"
    )

    monkeypatch.chdir(tmp_path)

    result = data_loader.load_data("PremierLeague")

    assert all(isinstance(team, str) for team in result["HomeTeam"])
    assert all(isinstance(team, str) for team in result["AwayTeam"])
    assert result.loc[0, "HomeTeam"] == "123"
    assert result.loc[0, "AwayTeam"] == "456"


def test_load_data_sorts_matches_chronologically(tmp_path, monkeypatch):
    """Test that matches are sorted by date."""

    league_path = tmp_path / "PremierLeague"
    league_path.mkdir()

    csv_file = league_path / "2023.csv"

    csv_file.write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
        "20/08/2023,Team C,Team D,1,0,H\n"
        "10/08/2023,Team A,Team B,2,1,H\n"
        "15/08/2023,Team E,Team F,0,0,D\n"
    )

    monkeypatch.chdir(tmp_path)

    result = data_loader.load_data("PremierLeague")

    assert list(result["Date"]) == [
        pd.Timestamp("2023-08-10"),
        pd.Timestamp("2023-08-15"),
        pd.Timestamp("2023-08-20"),
    ]


def test_load_data_resets_index(tmp_path, monkeypatch):
    """Test that the index is reset after sorting."""

    league_path = tmp_path / "PremierLeague"
    league_path.mkdir()

    csv_file = league_path / "2023.csv"

    csv_file.write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
        "20/08/2023,Team C,Team D,1,0,H\n"
        "10/08/2023,Team A,Team B,2,1,H\n"
    )

    monkeypatch.chdir(tmp_path)

    result = data_loader.load_data("PremierLeague")

    assert list(result.index) == [0, 1]


@pytest.mark.parametrize(
    "validation_seasons, expected_train, expected_validation",
    [
        (1, ["2022/2023", "2023/2024"], ["2024/2025"]),
        (2, ["2022/2023"], ["2023/2024", "2024/2025"]),
    ],
)
def test_split_by_season(
    validation_seasons,
    expected_train,
    expected_validation,
):
    """Test that data is split chronologically by season."""

    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2022-08-01",
                    "2023-01-01",
                    "2023-08-01",
                    "2024-01-01",
                    "2024-08-01",
                ]
            ),
            "HomeTeam": ["A", "B", "C", "D", "E"],
            "AwayTeam": ["B", "C", "D", "E", "F"],
        }
    )

    train_data, validation_data = data_loader.split_by_season(
        dataframe,
        validation_seasons=validation_seasons,
    )

    assert train_data["Season"].unique().tolist() == expected_train
    assert validation_data["Season"].unique().tolist() == expected_validation


def test_split_by_season_sorts_data():
    """Test that split_by_season sorts the data chronologically."""

    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-08-01",
                    "2022-08-01",
                    "2023-08-01",
                ]
            ),
            "HomeTeam": ["C", "A", "B"],
            "AwayTeam": ["D", "B", "C"],
        }
    )

    train_data, validation_data = data_loader.split_by_season(
        dataframe,
        validation_seasons=1,
    )

    assert train_data.iloc[0]["Date"] == pd.Timestamp("2022-08-01")
    assert validation_data.iloc[0]["Date"] == pd.Timestamp("2024-08-01")


def test_split_by_season_adds_season_column():
    """Test that the Season column is created."""

    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2022-08-01",
                    "2023-01-01",
                    "2023-08-01",
                ]
            ),
        }
    )

    train_data, validation_data = data_loader.split_by_season(
        dataframe,
        validation_seasons=1,
    )

    assert "Season" in train_data.columns
    assert "Season" in validation_data.columns


def test_split_by_season_preserves_match_data():
    """Test that splitting does not remove match information."""

    dataframe = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2022-08-01",
                    "2023-08-01",
                ]
            ),
            "HomeTeam": ["Team A", "Team B"],
            "AwayTeam": ["Team B", "Team C"],
            "FTHG": [2, 1],
            "FTAG": [1, 0],
            "FTR": ["H", "H"],
        }
    )

    train_data, validation_data = data_loader.split_by_season(
        dataframe,
        validation_seasons=1,
    )

    assert train_data.iloc[0]["HomeTeam"] == "Team A"
    assert train_data.iloc[0]["FTHG"] == 2

    assert validation_data.iloc[0]["HomeTeam"] == "Team B"
    assert validation_data.iloc[0]["FTHG"] == 1
