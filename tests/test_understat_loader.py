"""
Tests for the Understat data loader.
"""

import pandas as pd
import pytest

from src import understat_loader
from src.understat_client import UnderstatUnavailable


def season_frame(home="Manchester City", away="Arsenal", home_xg=1.8, away_xg=0.9,
                 day="2023-08-12"):
    """
    Builds a frame shaped like the client's output for one season.
    """
    return pd.DataFrame([{
        "date": pd.Timestamp(day),
        "home_team": home,
        "away_team": away,
        "home_xg": home_xg,
        "away_xg": away_xg,
    }])


def install(monkeypatch, results):
    """
    Stubs the client so each season returns a queued result or raises.

    Parameters:
        monkeypatch: The pytest monkeypatch fixture.
        results (dict): Season label to either a frame or an exception.
    """
    requested = []

    def fake_get(slug, season):
        requested.append(season)

        outcome = results[season]

        if isinstance(outcome, Exception):
            raise outcome

        return outcome

    monkeypatch.setattr(understat_loader, "get_league_season", fake_get)

    return requested


def test_load_returns_every_requested_season(monkeypatch):
    """A range of seasons comes back as one frame, newest last."""
    install(monkeypatch, {
        "2020/2021": season_frame(day="2020-08-15"),
        "2021/2022": season_frame(day="2021-08-14"),
        "2022/2023": season_frame(day="2022-08-13"),
    })

    result = understat_loader.load_understat_data("EPL", 2020, 2022)

    assert len(result) == 3
    assert list(result["date"]) == [
        pd.Timestamp("2020-08-15"),
        pd.Timestamp("2021-08-14"),
        pd.Timestamp("2022-08-13"),
    ]


def test_load_asks_for_each_season_on_its_own(monkeypatch):
    """Seasons are requested individually, so one gap cannot hide the rest.

    A single combined request is answered as a unit: if Understat has not
    published the newest season, the whole range comes back short and the
    missing data is only visible by comparing row counts.
    """
    requested = install(monkeypatch, {
        "2020/2021": season_frame(day="2020-08-15"),
        "2021/2022": season_frame(day="2021-08-14"),
        "2022/2023": season_frame(day="2022-08-13"),
    })

    understat_loader.load_understat_data("EPL", 2020, 2022)

    assert requested == ["2020/2021", "2021/2022", "2022/2023"]


def test_load_keeps_the_seasons_that_are_available(monkeypatch):
    """An unpublished season is skipped and named, not silently dropped."""

    install(monkeypatch, {
        "2024/2025": season_frame(day="2024-08-17"),
        "2025/2026": season_frame(day="2025-08-16"),
        "2026/2027": UnderstatUnavailable("no played matches"),
    })

    result = understat_loader.load_understat_data("EPL", 2024, 2026)

    assert len(result) == 2
    assert pd.Timestamp("2026-08-21") not in list(result["date"])


def test_load_reports_the_season_it_had_to_skip(monkeypatch, capsys):
    """The season label is printed so the gap is visible in the output."""
    install(monkeypatch, {
        "2025/2026": season_frame(day="2025-08-16"),
        "2026/2027": UnderstatUnavailable("no played matches"),
    })

    understat_loader.load_understat_data("EPL", 2025, 2026)

    assert "2026/2027" in capsys.readouterr().out


def test_load_raises_when_no_season_has_data(monkeypatch):
    """An entirely uncovered range is an error the caller can act on."""
    install(monkeypatch, {
        season: UnderstatUnavailable("no played matches")
        for season in ("2025/2026", "2026/2027")
    })

    with pytest.raises(understat_loader.UnderstatDataError, match="EPL"):
        understat_loader.load_understat_data("EPL", 2025, 2026)


def test_load_error_names_every_season_it_tried(monkeypatch):
    """The failure explains itself rather than just reporting nothing."""
    install(monkeypatch, {
        season: UnderstatUnavailable(f"{season} missing")
        for season in ("2024/2025", "2025/2026")
    })

    with pytest.raises(understat_loader.UnderstatDataError) as caught:
        understat_loader.load_understat_data("EPL", 2024, 2025)

    message = str(caught.value)

    assert "2024/2025 missing" in message
    assert "2025/2026 missing" in message


def test_load_translates_the_team_names(monkeypatch):
    """Understat's spelling is rewritten to the football-data.co.uk one."""
    install(monkeypatch, {
        "2025/2026": season_frame(home="Wolverhampton Wanderers"),
    })

    result = understat_loader.load_understat_data("EPL", 2025, 2025)

    assert result.loc[0, "home_team"] == "Wolves"


def test_load_returns_the_seasons_in_date_order(monkeypatch):
    """Frames are sorted after concatenation, not left in request order."""
    install(monkeypatch, {
        "2025/2026": season_frame(day="2026-05-01"),
        "2024/2025": season_frame(day="2024-08-01"),
    })

    result = understat_loader.load_understat_data("EPL", 2024, 2025)

    assert result["date"].is_monotonic_increasing
