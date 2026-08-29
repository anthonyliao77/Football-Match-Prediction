"""
Tests for the ELO rating system.
"""

import pandas as pd

# pyrefly: ignore [missing-import]
import pytest

from src import elo


@pytest.mark.parametrize(
    "date, expected",
    [
        (pd.Timestamp("2022-07-01"), "2022/2023"),
        (pd.Timestamp("2023-01-01"), "2022/2023"),
        (pd.Timestamp("2023-07-01"), "2023/2024"),
        (pd.Timestamp("2024-01-01"), "2023/2024"),
    ],
)
def test_get_season(date, expected):
    """Test season assignment around the July season boundary."""

    assert elo.get_season(date) == expected


@pytest.mark.parametrize(
    "result, home, expected",
    [
        ("H", True, 1),
        ("H", False, 0),
        ("D", True, 0.5),
        ("D", False, 0.5),
        ("A", True, 0),
        ("A", False, 1),
    ],
)
def test_get_match_result(result, home, expected):
    """Test match result encoding for home and away teams."""

    assert elo.get_match_result(result, home) == expected


@pytest.mark.parametrize(
    "home_rating, away_rating, expected",
    [
        (1500, 1500, 0.5),
        (1600, 1500, 0.640064999),
        (1500, 1600, 0.359935001),
    ],
)
def test_calculate_expected_score(
    home_rating,
    away_rating,
    expected,
):
    """Test expected score for different ELO rating combinations."""

    assert elo.calculate_expected_score(
        home_rating,
        away_rating,
    ) == pytest.approx(expected)


def test_calculate_expected_score_symmetry():
    """Test that expected scores for equal ratings sum to one."""

    home_score = elo.calculate_expected_score(1600, 1500)
    away_score = elo.calculate_expected_score(1500, 1600)

    assert home_score + away_score == pytest.approx(1.0)


@pytest.mark.parametrize(
    "rating, result, expected_score, expected",
    [
        (1500, 1, 0.5, 1516),
        (1500, 0, 0.5, 1484),
        (1500, 0.5, 0.5, 1500),
    ],
)
def test_update_elo_ratings(
    rating,
    result,
    expected_score,
    expected,
):
    """Test ELO rating updates for different match outcomes."""

    assert elo.update_elo_ratings(
        rating,
        result,
        expected_score,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "old_rating, expected",
    [
        (1500, 1500),
        (1600, 1575),
        (1400, 1425),
        (1800, 1725),
    ],
)
def test_new_season_elo_regression(old_rating, expected):
    """Test regression of ELO ratings toward the initial rating."""

    assert elo.new_season_elo_regression(
        old_rating
    ) == pytest.approx(expected)


def test_create_elo_features_initial_ratings():
    """Test that teams start with the initial ELO rating."""

    dataframe = pd.DataFrame(
        {
            "Date": [
                pd.Timestamp("2024-08-01"),
            ],
            "HomeTeam": ["Team A"],
            "AwayTeam": ["Team B"],
            "FTR": ["H"],
        }
    )

    result = elo.create_elo_features(dataframe)

    assert result.loc[0, "HomeEloBefore"] == pytest.approx(1500)
    assert result.loc[0, "AwayEloBefore"] == pytest.approx(1500)


def test_create_elo_features_updates_ratings():
    """Test that ELO ratings are updated after a match."""

    dataframe = pd.DataFrame(
        {
            "Date": [
                pd.Timestamp("2024-08-01"),
                pd.Timestamp("2024-08-08"),
            ],
            "HomeTeam": ["Team A", "Team A"],
            "AwayTeam": ["Team B", "Team B"],
            "FTR": ["H", "A"],
        }
    )

    result = elo.create_elo_features(dataframe)

    # First match starts with both teams at 1500
    assert result.loc[0, "HomeEloBefore"] == pytest.approx(1500)
    assert result.loc[0, "AwayEloBefore"] == pytest.approx(1500)

    # Second match must use the ratings produced by the first match
    assert result.loc[1, "HomeEloBefore"] == pytest.approx(
        result.loc[0, "HomeEloAfter"]
    )
    assert result.loc[1, "AwayEloBefore"] == pytest.approx(
        result.loc[0, "AwayEloAfter"]
    )


def test_create_elo_features_winner_rating_increases():
    """Test that the winning team's ELO increases."""

    dataframe = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-08-01")],
            "HomeTeam": ["Team A"],
            "AwayTeam": ["Team B"],
            "FTR": ["H"],
        }
    )

    result = elo.create_elo_features(dataframe)

    assert result.loc[0, "HomeEloAfter"] > result.loc[0, "HomeEloBefore"]
    assert result.loc[0, "AwayEloAfter"] < result.loc[0, "AwayEloBefore"]


def test_create_elo_features_away_winner():
    """Test that the away team's ELO increases after an away win."""

    dataframe = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-08-01")],
            "HomeTeam": ["Team A"],
            "AwayTeam": ["Team B"],
            "FTR": ["A"],
        }
    )

    result = elo.create_elo_features(dataframe)

    assert result.loc[0, "AwayEloAfter"] > result.loc[0, "AwayEloBefore"]
    assert result.loc[0, "HomeEloAfter"] < result.loc[0, "HomeEloBefore"]


def test_create_elo_features_draw():
    """Test ELO behavior after a draw."""

    dataframe = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-08-01")],
            "HomeTeam": ["Team A"],
            "AwayTeam": ["Team B"],
            "FTR": ["D"],
        }
    )

    result = elo.create_elo_features(dataframe)

    # Because home advantage makes the home team the favorite,
    # a draw should decrease the home rating and increase the away rating
    assert result.loc[0, "HomeEloAfter"] < result.loc[0, "HomeEloBefore"]
    assert result.loc[0, "AwayEloAfter"] > result.loc[0, "AwayEloBefore"]


def test_create_elo_features_uses_home_advantage():
    """Test that home advantage affects the expected result."""

    dataframe = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-08-01")],
            "HomeTeam": ["Team A"],
            "AwayTeam": ["Team B"],
            "FTR": ["H"],
        }
    )

    result = elo.create_elo_features(dataframe)

    # With equal ratings and a home advantage of 100,
    # the home team's expected score is greater than 0.5
    expected_home_score = elo.calculate_expected_score(
        1500 + elo.HOME_ADVANTAGE,
        1500,
    )

    expected_home_rating = elo.update_elo_ratings(
        1500,
        1,
        expected_home_score,
    )

    assert result.loc[0, "HomeEloAfter"] == pytest.approx(
        expected_home_rating
    )


def test_create_elo_features_no_current_match_leakage():
    """Test that pre-match ELO does not include the current match result."""

    first_match = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-08-01")],
            "HomeTeam": ["Team A"],
            "AwayTeam": ["Team B"],
            "FTR": ["H"],
        }
    )

    second_match = pd.DataFrame(
        {
            "Date": [
                pd.Timestamp("2024-08-01"),
                pd.Timestamp("2024-08-08"),
            ],
            "HomeTeam": ["Team A", "Team A"],
            "AwayTeam": ["Team B", "Team B"],
            "FTR": ["H", "A"],
        }
    )

    first_result = elo.create_elo_features(first_match)
    second_result = elo.create_elo_features(second_match)

    # The second match's pre-match ratings must be exactly the
    # ratings produced after the first match
    assert second_result.loc[1, "HomeEloBefore"] == pytest.approx(
        first_result.loc[0, "HomeEloAfter"]
    )
    assert second_result.loc[1, "AwayEloBefore"] == pytest.approx(
        first_result.loc[0, "AwayEloAfter"]
    )


def test_create_elo_features_season_regression():
    """Test that ELO ratings regress at the start of a new season."""

    dataframe = pd.DataFrame(
        {
            "Date": [
                pd.Timestamp("2024-08-01"),
                pd.Timestamp("2025-06-01"),
                pd.Timestamp("2025-08-01"),
            ],
            "HomeTeam": ["Team A", "Team A", "Team A"],
            "AwayTeam": ["Team B", "Team B", "Team B"],
            "FTR": ["H", "H", "H"],
        }
    )

    result = elo.create_elo_features(dataframe)

    # The first match of the new season should have ratings
    # regressed toward 1500
    previous_rating = result.loc[1, "HomeEloAfter"]

    expected_regressed_rating = (
        elo.A_FACTOR * previous_rating
        + (1 - elo.A_FACTOR) * elo.INITIAL_RATING
    )

    assert result.loc[2, "HomeEloBefore"] == pytest.approx(
        expected_regressed_rating
    )


def test_create_elo_features_multiple_teams():
    """Test that separate team ratings are maintained independently."""

    dataframe = pd.DataFrame(
        {
            "Date": [
                pd.Timestamp("2024-08-01"),
                pd.Timestamp("2024-08-02"),
            ],
            "HomeTeam": ["Team A", "Team C"],
            "AwayTeam": ["Team B", "Team D"],
            "FTR": ["H", "A"],
        }
    )

    result = elo.create_elo_features(dataframe)

    # All teams should begin with the initial rating because
    # they have not played before
    assert result.loc[0, "HomeEloBefore"] == pytest.approx(1500)
    assert result.loc[0, "AwayEloBefore"] == pytest.approx(1500)
    assert result.loc[1, "HomeEloBefore"] == pytest.approx(1500)
    assert result.loc[1, "AwayEloBefore"] == pytest.approx(1500)


def test_create_elo_features_adds_elo_columns():
    """Test that ELO feature columns are added to the DataFrame."""

    dataframe = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-08-01")],
            "HomeTeam": ["Team A"],
            "AwayTeam": ["Team B"],
            "FTR": ["H"],
        }
    )

    result = elo.create_elo_features(dataframe)

    expected_columns = {
        "HomeEloBefore",
        "AwayEloBefore",
        "HomeEloAfter",
        "AwayEloAfter",
    }

    assert expected_columns.issubset(result.columns)