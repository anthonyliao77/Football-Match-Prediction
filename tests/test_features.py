"""
Tests for rolling match features.
"""

import pandas as pd

# pyrefly: ignore [missing-import]
import pytest

from src import features


@pytest.mark.parametrize(
    "result, home, expected",
    [
        ("H", True, 3),
        ("H", False, 0),
        ("D", True, 1),
        ("D", False, 1),
        ("A", True, 0),
        ("A", False, 3),
    ],
)
def test_calculate_points(result, home, expected):
    """Test points earned from different match results."""

    assert features.calculate_points(result, home) == expected


@pytest.mark.parametrize(
    "team_points, expected",
    [
        ([], 0),
        ([{"points": 3}], 3),
        ([{"points": 3}, {"points": 1}, {"points": 0}], 4),
        (
            [
                {"points": 3},
                {"points": 3},
                {"points": 3},
                {"points": 3},
                {"points": 3},
            ],
            15,
        ),
    ],
)
def test_last5_points(team_points, expected):
    """Test calculation of total points from previous matches."""

    assert features.last5_points(team_points) == expected


@pytest.mark.parametrize(
    "team_goals, expected",
    [
        ([], 0),
        ([{"goals_scored": 2}], 2),
        (
            [
                {"goals_scored": 2},
                {"goals_scored": 1},
                {"goals_scored": 3},
            ],
            6,
        ),
    ],
)
def test_last5_goal_scored(team_goals, expected):
    """Test calculation of goals scored from previous matches."""

    assert features.last5_goal_scored(team_goals) == expected


@pytest.mark.parametrize(
    "team_goals, expected",
    [
        ([], 0),
        ([{"goals_conceded": 2}], 2),
        (
            [
                {"goals_conceded": 2},
                {"goals_conceded": 1},
                {"goals_conceded": 3},
            ],
            6,
        ),
    ],
)
def test_last5_goal_conceded(team_goals, expected):
    """Test calculation of goals conceded from previous matches."""

    assert features.last5_goal_conceded(team_goals) == expected


@pytest.mark.parametrize(
    "goals_scored, goals_conceded, expected",
    [
        (0, 0, 0),
        (5, 2, 3),
        (2, 5, -3),
        (10, 10, 0),
    ],
)
def test_last5_goal_difference(
    goals_scored,
    goals_conceded,
    expected,
):
    """Test calculation of goal difference."""

    assert features.last5_goal_difference(
        goals_scored,
        goals_conceded,
    ) == expected


@pytest.mark.parametrize(
    "team_shots_on_target, expected",
    [
        ([], 0),
        ([{"shots_on_target": 5}], 5),
        (
            [
                {"shots_on_target": 5},
                {"shots_on_target": 3},
                {"shots_on_target": 7},
            ],
            15,
        ),
    ],
)
def test_last5_shots_on_target(team_shots_on_target, expected):
    """Test calculation of shots on target from previous matches."""

    assert features.last5_shots_on_target(
        team_shots_on_target
    ) == expected


@pytest.mark.parametrize(
    "team_shots, expected",
    [
        ([], 0),
        ([{"shots": 10}], 10),
        (
            [
                {"shots": 10},
                {"shots": 8},
                {"shots": 12},
            ],
            30,
        ),
    ],
)
def test_last5_shots(team_shots, expected):
    """Test calculation of total shots from previous matches."""

    assert features.last5_shots(team_shots) == expected


@pytest.mark.parametrize(
    "shots, goals, expected",
    [
        (0, 0, 0.0),
        (10, 0, 0.0),
        (10, 2, 0.2),
        (20, 5, 0.25),
        (8, 8, 1.0),
    ],
)
def test_last5_shots_conversion(shots, goals, expected):
    """Test shots conversion calculation."""

    assert features.last5_shots_conversion(
        shots,
        goals,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "team_xg, expected",
    [
        ([], 0),
        ([{"xG": 1.5}], 1.5),
        (
            [
                {"xG": 1.2},
                {"xG": 0.8},
                {"xG": 2.1},
            ],
            4.1,
        ),
    ],
)
def test_last5_xg(team_xg, expected):
    """Test calculation of total expected goals from previous matches."""

    assert features.last5_xg(team_xg) == pytest.approx(expected)


def test_create_features_initial_match():
    """Test that the first match has no previous-match features."""

    dataframe = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-08-01")],
            "HomeTeam": ["Team A"],
            "AwayTeam": ["Team B"],
            "FTHG": [2],
            "FTAG": [1],
            "HST": [5],
            "AST": [3],
            "HS": [10],
            "AS": [8],
            "home_xg": [1.8],
            "away_xg": [0.9],
            "FTR": ["H"],
        }
    )

    result = features.create_features(dataframe)

    assert result.loc[0, "HomePT5"] == 0
    assert result.loc[0, "AwayPT5"] == 0
    assert result.loc[0, "HomeGS5"] == 0
    assert result.loc[0, "AwayGS5"] == 0
    assert result.loc[0, "HomeGC5"] == 0
    assert result.loc[0, "AwayGC5"] == 0
    assert result.loc[0, "HomeGD5"] == 0
    assert result.loc[0, "AwayGD5"] == 0
    assert result.loc[0, "HomeSOT5"] == 0
    assert result.loc[0, "AwaySOT5"] == 0
    assert result.loc[0, "HomeS5"] == 0
    assert result.loc[0, "AwayS5"] == 0
    assert result.loc[0, "HomeSC5"] == 0
    assert result.loc[0, "AwaySC5"] == 0
    assert result.loc[0, "HomeXG5"] == 0
    assert result.loc[0, "AwayXG5"] == 0
    assert result.loc[0, "HomeXGA5"] == 0
    assert result.loc[0, "AwayXGA5"] == 0


def test_create_features_only_uses_last_five_matches():
    """Test that only the five most recent matches are used."""

    dates = pd.date_range(
        start="2024-08-01",
        periods=7,
        freq="7D",
    )

    dataframe = pd.DataFrame(
        {
            "Date": dates,
            "HomeTeam": ["Team A"] * 7,
            "AwayTeam": ["Team B"] * 7,
            "FTHG": [10, 2, 3, 4, 5, 6, 100],
            "FTAG": [0, 0, 0, 0, 0, 0, 0],
            "HST": [10, 2, 3, 4, 5, 6, 100],
            "AST": [0, 0, 0, 0, 0, 0, 0],
            "HS": [10, 2, 3, 4, 5, 6, 100],
            "AS": [1, 1, 1, 1, 1, 1, 100],
            "home_xg": [10.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0],
            "away_xg": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "FTR": ["H", "H", "H", "H", "H", "H", "H"],
        }
    )

    result = features.create_features(dataframe)

    # The seventh match should use matches 2-6:
    # the five most recent completed matches.
    # Match 1 (index 0) is too old and must be excluded.
    # Match 7 (index 6) is the current match and is not included.
    assert result.loc[6, "HomeGS5"] == 20
    assert result.loc[6, "HomePT5"] == 15
    assert result.loc[6, "HomeS5"] == 20
    assert result.loc[6, "HomeSOT5"] == 20
    assert result.loc[6, "HomeXG5"] == pytest.approx(20.0)


def test_create_features_maintains_separate_team_histories():
    """Test that each team's rolling history is maintained independently."""

    dataframe = pd.DataFrame(
        {
            "Date": [
                pd.Timestamp("2024-08-01"),
                pd.Timestamp("2024-08-02"),
                pd.Timestamp("2024-08-03"),
            ],
            "HomeTeam": ["Team A", "Team C", "Team A"],
            "AwayTeam": ["Team B", "Team D", "Team B"],
            "FTHG": [2, 5, 3],
            "FTAG": [1, 0, 2],
            "HST": [5, 8, 6],
            "AST": [3, 2, 4],
            "HS": [10, 12, 11],
            "AS": [8, 5, 9],
            "home_xg": [1.8, 3.0, 2.2],
            "away_xg": [0.9, 0.5, 1.1],
            "FTR": ["H", "H", "H"],
        }
    )

    result = features.create_features(dataframe)

    # Team A's third match should only contain Team A's first match.
    assert result.loc[2, "HomeGS5"] == 2
    assert result.loc[2, "HomeGC5"] == 1
    assert result.loc[2, "HomePT5"] == 3

    # Team B's third match should only contain Team B's first match.
    assert result.loc[2, "AwayGS5"] == 1
    assert result.loc[2, "AwayGC5"] == 2
    assert result.loc[2, "AwayPT5"] == 0


def test_create_features_does_not_use_current_match():
    """Test that current match statistics are excluded from features."""

    dataframe = pd.DataFrame(
        {
            "Date": [
                pd.Timestamp("2024-08-01"),
                pd.Timestamp("2024-08-08"),
            ],
            "HomeTeam": ["Team A", "Team A"],
            "AwayTeam": ["Team B", "Team B"],
            "FTHG": [2, 100],
            "FTAG": [1, 100],
            "HST": [5, 100],
            "AST": [3, 100],
            "HS": [10, 100],
            "AS": [8, 100],
            "home_xg": [1.8, 100.0],
            "away_xg": [0.9, 100.0],
            "FTR": ["H", "A"],
        }
    )

    result = features.create_features(dataframe)

    # The second match should only reflect the first match.
    assert result.loc[1, "HomeGS5"] == 2
    assert result.loc[1, "AwayGS5"] == 1
    assert result.loc[1, "HomeS5"] == 10
    assert result.loc[1, "AwayS5"] == 8
    assert result.loc[1, "HomeXG5"] == pytest.approx(1.8)
    assert result.loc[1, "AwayXG5"] == pytest.approx(0.9)


def test_create_features_first_match_has_no_history():
    """Test that the first match has no historical features."""

    dataframe = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-08-01")],
            "HomeTeam": ["Team A"],
            "AwayTeam": ["Team B"],
            "FTHG": [2],
            "FTAG": [1],
            "HST": [5],
            "AST": [3],
            "HS": [10],
            "AS": [8],
            "home_xg": [1.8],
            "away_xg": [0.9],
            "FTR": ["H"],
        }
    )

    result = features.create_features(dataframe)

    assert result.loc[0, "HomePT5"] == 0
    assert result.loc[0, "AwayPT5"] == 0

    assert result.loc[0, "HomeGS5"] == 0
    assert result.loc[0, "AwayGS5"] == 0

    assert result.loc[0, "HomeGC5"] == 0
    assert result.loc[0, "AwayGC5"] == 0

    assert result.loc[0, "HomeGD5"] == 0
    assert result.loc[0, "AwayGD5"] == 0

    assert result.loc[0, "HomeSOT5"] == 0
    assert result.loc[0, "AwaySOT5"] == 0

    assert result.loc[0, "HomeS5"] == 0
    assert result.loc[0, "AwayS5"] == 0

    assert result.loc[0, "HomeSC5"] == pytest.approx(0.0)
    assert result.loc[0, "AwaySC5"] == pytest.approx(0.0)

    assert result.loc[0, "HomeXG5"] == pytest.approx(0.0)
    assert result.loc[0, "AwayXG5"] == pytest.approx(0.0)

    assert result.loc[0, "HomeXGA5"] == pytest.approx(0.0)
    assert result.loc[0, "AwayXGA5"] == pytest.approx(0.0)

@pytest.mark.parametrize("gap", [None, float("nan"), pd.NA, ""])
def test_number_coerces_a_gap_to_zero(gap):
    """Every spelling of "no value" becomes 0.0 rather than a NaN."""

    assert features._number(gap) == 0.0


def test_number_passes_through_a_real_value():
    """A present number is read as a float, not rounded or stringified."""

    assert features._number(3) == 3.0
    assert features._number("2.5") == 2.5
    assert features._number(1.75) == 1.75


def test_number_survives_nonsense():
    """A value that is not a number at all is treated as missing."""

    assert features._number("n/a") == 0.0


def test_create_features_survives_missing_shots_and_xg():
    """A gap in the shot or xG columns must not poison the rolling sums.

    Every feature is a sum over a team's previous matches, so a single missing
    value would turn the whole column into NaN. The sum would not fail, it
    would pass NaN to the model, and the failure would surface at predict time
    with nothing pointing back at the CSV cell that caused it.
    """
    dates = pd.date_range(start="2024-08-01", periods=4, freq="7D")

    dataframe = pd.DataFrame(
        {
            "Date": dates,
            "HomeTeam": ["Team A"] * 4,
            "AwayTeam": ["Team B"] * 4,
            "FTHG": [2, None, 1, 3],
            "FTAG": [1, 0, None, 0],
            "HST": [5, None, 4, 6],
            "AST": [3, 2, None, 1],
            "HS": [10, 12, None, 15],
            "AS": [8, 7, 9, None],
            "home_xg": [1.8, None, 1.1, 2.4],
            "away_xg": [0.9, 1.2, None, 0.6],
            "FTR": ["H", "H", "D", "H"],
        }
    )

    result = features.create_features(dataframe)

    numeric = result.select_dtypes(include="number")

    assert not numeric.isna().any().any()

    for column in [
        "HomeGS5", "AwayGS5", "HomeGC5", "AwayGC5",
        "HomeS5", "AwayS5", "HomeSOT5", "AwaySOT5",
        "HomeXG5", "AwayXG5", "HomeXGA5", "AwayXGA5",
    ]:
        assert result[column].notna().all(), column


def test_create_features_sums_only_the_values_that_are_present():
    """A missing value contributes nothing rather than wiping the window."""

    dates = pd.date_range(start="2024-08-01", periods=4, freq="7D")

    dataframe = pd.DataFrame(
        {
            "Date": dates,
            "HomeTeam": ["Team A"] * 4,
            "AwayTeam": ["Team B"] * 4,
            "FTHG": [2, 1, 3, 1],
            "FTAG": [1, 0, 0, 2],
            "HST": [5, 4, 6, 2],
            "AST": [3, 2, 1, 3],
            "HS": [10, 12, 15, 9],
            "AS": [8, 7, 9, 11],
            # The second match is missing xG for the home side.
            "home_xg": [1.0, None, 2.0, 3.0],
            "away_xg": [0.5, 0.5, 0.5, 0.5],
            "FTR": ["H", "H", "H", "H"],
        }
    )

    result = features.create_features(dataframe)

    # Row 2's window is rows 0 and 1: the gap counts as 0, giving 1.0 rather
    # than a NaN that would take the whole column with it.
    assert result.loc[2, "HomeXG5"] == pytest.approx(1.0)

    # Row 3's window is rows 0, 1 and 2: 1.0 + 0.0 + 2.0. Its own 3.0 is
    # excluded, which is the whole point of a pre-match feature.
    assert result.loc[3, "HomeXG5"] == pytest.approx(3.0)

    assert result.loc[3, "AwayXG5"] == pytest.approx(1.5)
