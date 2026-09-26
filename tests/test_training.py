"""
Tests for the xG coverage reporting in the training pipeline.

The pipeline no longer fills missing xG, so the only thing standing between a
season Understat has not published and a set of quietly meaningless scores is
this report. These tests cover it.
"""

import pandas as pd
import pytest

from config import FEATURE_COLUMNS
from src.elo import create_elo_features
from src.features import create_features
from src.training import _warn_on_missing_xg, _xg_coverage_by_season, train_model


def build_seasons(per_season=12, covered=("2024/2025",), gap=()):
    """
    Builds a frame of seasons, some with xG and some without.

    Parameters:
        per_season (int): Matches per season.
        covered (tuple): Season labels that carry xG.
        gap (tuple): Season labels that do not.
    """
    frames = []
    start_years = {season: int(season.split("/")[0]) for season in covered + gap}

    for season, start_year in start_years.items():
        dates = pd.date_range(f"{start_year}-08-10", periods=per_season, freq="7D")
        has_xg = season in covered

        records = []

        for index, date in enumerate(dates):
            record = {
                "Date": date,
                "HomeTeam": f"H{index % 4}",
                "AwayTeam": f"A{index % 4}",
                "HS": 10 + index,
                "AS": 8 + index,
                "HST": 4 + index % 3,
                "AST": 2 + index % 3,
                "FTHG": index % 4,
                "FTAG": (index + 1) % 3,
                "FTR": ["H", "D", "A"][index % 3],
                "home_xg": 1.0 + index if has_xg else None,
                "away_xg": 0.5 + index if has_xg else None,
            }

            records.append(record)

        frames.append(pd.DataFrame(records))

    return pd.concat(frames, ignore_index=True)


def test_coverage_counts_rows_and_gaps_per_season():
    """Each season is accounted for separately."""
    frame = build_seasons(
        covered=("2024/2025",),
        gap=("2025/2026", "2026/2027"),
    )

    coverage = _xg_coverage_by_season(frame)

    assert list(coverage.index) == ["2024/2025", "2025/2026", "2026/2027"]
    assert coverage.loc["2024/2025", "gaps"] == 0
    assert coverage.loc["2026/2027", "gaps"] == 12
    assert coverage["rows"].sum() == 36


def test_coverage_treats_one_missing_side_as_a_gap():
    """A row with only one of the two values is not fully covered."""
    frame = build_seasons(covered=("2024/2025",))
    frame.loc[0, "away_xg"] = None

    coverage = _xg_coverage_by_season(frame)

    assert coverage.loc["2024/2025", "gaps"] == 1


def test_no_warning_when_the_newest_season_is_covered(capsys):
    """A complete season produces one coverage line and no alarm."""
    frame = build_seasons(
        covered=("2024/2025", "2025/2026"),
        gap=(),
    )

    _warn_on_missing_xg(frame, "PremierLeague")

    output = capsys.readouterr().out

    assert "24 of 24 rows carry measured xG" in output
    assert "WARNING" not in output


def test_warning_fires_for_an_uncovered_current_season(capsys):
    """The season being validated on is the one that has to be covered."""
    frame = build_seasons(
        covered=("2024/2025",),
        gap=("2025/2026", "2026/2027"),
    )

    _warn_on_missing_xg(frame, "PremierLeague")

    output = capsys.readouterr().out

    assert "WARNING" in output
    assert "12 of 12 rows in 2026/2027 have no measured xG" in output


def test_warning_names_the_four_features_and_the_fix(capsys):
    """The message has to say what is wrong and what to run about it."""
    frame = build_seasons(covered=("2024/2025",), gap=("2026/2027",))

    _warn_on_missing_xg(frame, "SerieA")

    output = capsys.readouterr().out

    for feature in ["HomeXG5", "AwayXG5", "HomeXGA5", "AwayXGA5"]:
        assert feature in output

    assert "backfill_xg.py --league SerieA" in output


def test_warning_does_not_fire_for_an_older_gap(capsys):
    """A hole in an old season does not affect the scores being reported."""
    frame = build_seasons(
        covered=("2024/2025", "2026/2027"),
        gap=("2025/2026",),
    )

    _warn_on_missing_xg(frame, "PremierLeague")

    output = capsys.readouterr().out

    assert "WARNING" not in output
    assert "24 of 36 rows carry measured xG" in output


def test_missing_xg_reaches_the_features_as_zero():
    """The consequence the warning describes, stated as a test.

    create_features sums each team's previous xG, and a missing value is
    coerced to zero rather than poisoning the column. That is what keeps the
    run going, and it is also why a silent gap would be so easy to miss.
    """
    # A single uncovered season, so no team has any earlier match to draw on.
    # This is the early-August case the warning exists for.
    frame = build_seasons(per_season=4, covered=(), gap=("2026/2027",))

    result = create_elo_features(dataframe=create_features(dataframe=frame))

    for column in FEATURE_COLUMNS:
        assert result[column].notna().all(), column

    assert (result["HomeXG5"] == 0).all()
    assert (result["HomeXGA5"] == 0).all()
    assert (result["AwayXG5"] == 0).all()
    assert (result["AwayXGA5"] == 0).all()


def test_train_model_rejects_an_unknown_league():
    """A typo in the league name fails with the list of valid ones."""
    with pytest.raises(ValueError, match="Available leagues"):
        train_model("Premier League")
