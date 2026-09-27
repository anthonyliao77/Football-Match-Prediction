"""
Tests for the refresh script.

The script itself holds no fetching logic, so what is worth testing is the
behaviour that comes from orchestrating: that both steps are attempted, that one
league failing does not cost the others, that the exit status says so, and that
the number it leads with is the one a person running it needs, which is how far
behind their results download is.
"""

import pandas as pd
import pytest

import refresh_data
from add_fixtures import UnderstatUnavailable
from backfill_xg import UnderstatDataError
from config import LEAGUES


def argparse_namespace(league="all", dry_run=False):
    """Builds a stand-in for the parsed command line."""
    import argparse

    return argparse.Namespace(league=league, season=None, dry_run=dry_run)


@pytest.fixture
def league_dirs(tmp_path, monkeypatch):
    """
    Builds a season CSV per league in a temporary location.

    The script resolves its paths through config, so the tests chdir rather than
    patch the config, which keeps the real path logic under test.
    """
    for league in LEAGUES:
        directory = tmp_path / LEAGUES[league]["football_data"]
        directory.mkdir(parents=True)

        pd.DataFrame([{
            "Date": "12/08/2026",
            "Div": "E0",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Chelsea",
            "FTHG": 2,
            "FTAG": 1,
            "FTR": "H",
        }]).to_csv(directory / "2026-2027.csv", index=False)

    monkeypatch.chdir(tmp_path)

    return tmp_path


def stub(monkeypatch, fixtures=None, xg=None, fixtures_error=None, xg_error=None):
    """
    Replaces both network-facing steps with recorded counts.
    """
    calls = {"fixtures": [], "xg": []}

    def run_fixtures(league, season, dry_run=False):
        calls["fixtures"].append((league, season, dry_run))

        if fixtures_error is not None:
            raise fixtures_error

        return {
            "added": 3,
            "already_present": 10,
            "rescheduled": 1,
            "needs_results": 4,
            "unknown_teams": [],
        }

    def run_xg(league, dry_run=False):
        calls["xg"].append((league, dry_run))

        if xg_error is not None:
            raise xg_error

        return {
            "files": 2,
            "rows": 700,
            "xg_filled": 12,
            "xg_already_present": 688,
            "rows_unmatched": 0,
        }

    monkeypatch.setattr(refresh_data, "add_fixtures", run_fixtures)
    monkeypatch.setattr(refresh_data, "backfill_league_xg", run_xg)

    return calls


def test_every_league_is_refreshed(monkeypatch, league_dirs):
    """The default is all three leagues, not just the first."""
    calls = stub(monkeypatch)

    rows = [
        refresh_data.refresh_league(league, None, dry_run=False)
        for league in LEAGUES
    ]

    assert [row["league"] for row in rows] == list(LEAGUES)
    assert [call[0] for call in calls["fixtures"]] == list(LEAGUES)


def test_both_steps_run_for_a_league(monkeypatch, league_dirs):
    """Fixtures and xG are both called, xG after the fixtures."""
    order = []

    monkeypatch.setattr(
        refresh_data,
        "add_fixtures",
        lambda league, season, dry_run=False: order.append("fixtures")
        or {
            "added": 0,
            "rescheduled": 0,
            "needs_results": 0,
            "unknown_teams": [],
        },
    )

    monkeypatch.setattr(
        refresh_data,
        "backfill_league_xg",
        lambda league, dry_run=False: order.append("xg")
        or {"xg_filled": 0, "rows": 0, "rows_unmatched": 0},
    )

    refresh_data.refresh_league("PremierLeague", None, dry_run=False)

    assert order == ["fixtures", "xg"]


def test_the_newest_season_is_used_when_none_is_given(monkeypatch, league_dirs):
    """An unqualified run fills the season the file already holds."""
    calls = stub(monkeypatch)

    refresh_data.refresh_league("PremierLeague", None, dry_run=False)

    assert calls["fixtures"][0][1] == "2026/2027"


def test_the_xg_step_ignores_the_season_flag(monkeypatch, league_dirs):
    """xG fills empty cells, so narrowing the season would only hide gaps.

    Asking for one season's fixtures and then backfilling only that season
    would leave an empty xG cell in an older season looking like a deliberate
    absence rather than a gap nobody filled.
    """
    calls = stub(monkeypatch)

    refresh_data.refresh_league("PremierLeague", "2026/2027", dry_run=False)

    assert calls["fixtures"][0][1] == "2026/2027"
    assert calls["xg"] == [("PremierLeague", False)]


def test_dry_run_reaches_both_steps(monkeypatch, league_dirs):
    """A preview that skipped a step would not be a preview of the run."""
    calls = stub(monkeypatch)

    refresh_data.refresh_league("PremierLeague", None, dry_run=True)

    assert calls["fixtures"][0][2] is True
    assert calls["xg"][0][1] is True


def test_one_league_failing_does_not_stop_the_others(monkeypatch, league_dirs):
    """Understat being briefly unreadable should not cost two leagues.

    It is an undocumented endpoint with no published availability, so a failure
    is a normal event rather than an exceptional one, and the other leagues are
    independent of it.
    """
    def run_fixtures(league, season, dry_run=False):
        if league == "LaLiga":
            raise UnderstatUnavailable("Understat has no fixtures listed for SP1")

        return {
            "added": 1,
            "rescheduled": 0,
            "needs_results": 0,
            "unknown_teams": [],
        }

    monkeypatch.setattr(refresh_data, "add_fixtures", run_fixtures)

    monkeypatch.setattr(
        refresh_data,
        "backfill_league_xg",
        lambda league, dry_run=False: {"xg_filled": 5, "rows": 1, "rows_unmatched": 0},
    )

    rows = [
        refresh_data.refresh_league(league, None, dry_run=False)
        for league in LEAGUES
    ]

    assert rows[1]["error"] is not None
    assert rows[0]["error"] is None
    assert rows[2]["error"] is None
    assert rows[0]["added"] == 1
    assert rows[2]["xg_filled"] == 5


def test_a_failed_fixtures_step_skips_xg_for_that_league(
    monkeypatch, league_dirs
):
    """There is no point filling xG for a season whose fixtures never arrived."""
    calls = stub(
        monkeypatch,
        fixtures_error=UnderstatUnavailable("nothing listed"),
    )

    row = refresh_data.refresh_league("PremierLeague", None, dry_run=False)

    assert row["error"]
    assert calls["xg"] == []


def test_a_failed_xg_step_still_reports_the_fixtures(monkeypatch, league_dirs):
    """The fixture half of the work succeeded and should be counted."""
    stub(monkeypatch, xg_error=UnderstatDataError("no Understat data"))

    row = refresh_data.refresh_league("PremierLeague", None, dry_run=False)

    assert row["added"] == 3
    assert row["rescheduled"] == 1
    assert "xG" in row["error"]


def test_a_league_with_no_season_file_is_reported(monkeypatch, league_dirs):
    """Nothing to refresh is stated, not silently skipped."""
    calls = stub(monkeypatch)

    (league_dirs / LEAGUES["SerieA"]["football_data"] / "2026-2027.csv").unlink()

    row = refresh_data.refresh_league("SerieA", None, dry_run=False)

    assert "no season CSVs" in row["error"]
    assert calls["fixtures"] == []


def test_the_exit_status_is_non_zero_when_a_league_fails(
    monkeypatch, league_dirs
):
    """This is what makes the command safe to put on a schedule."""
    def run_fixtures(league, season, dry_run=False):
        if league == "SerieA":
            raise UnderstatUnavailable("nothing listed")

        return {
            "added": 0,
            "rescheduled": 0,
            "needs_results": 0,
            "unknown_teams": [],
        }

    monkeypatch.setattr(refresh_data, "add_fixtures", run_fixtures)

    monkeypatch.setattr(
        refresh_data,
        "backfill_league_xg",
        lambda league, dry_run=False: {"xg_filled": 0, "rows": 0, "rows_unmatched": 0},
    )

    monkeypatch.setattr(
        refresh_data, "parse_arguments", lambda: argparse_namespace()
    )

    assert refresh_data.main() == 1

    monkeypatch.setattr(refresh_data, "parse_arguments", lambda: argparse_namespace("PremierLeague"))

    assert refresh_data.main() == 0


def test_a_clean_run_exits_zero(monkeypatch, league_dirs):
    """A league that cannot be reached is a failure, not a warning."""
    stub(monkeypatch)

    monkeypatch.setattr(refresh_data, "parse_arguments", lambda: argparse_namespace())

    assert refresh_data.main() == 0


def test_the_stale_count_is_the_headline(monkeypatch, league_dirs, capsys):
    """The number the person running it needs is how far behind they are.

    Neither underlying script can report this. A fixture added before kickoff
    sits in the file with a blank result, matches on date and sides, and is
    counted as already present by both of them, so nothing anywhere says the
    results download is behind. The refresh exists largely to say it.
    """
    stub(monkeypatch)

    monkeypatch.setattr(refresh_data, "parse_arguments", lambda: argparse_namespace())

    refresh_data.main()

    output = capsys.readouterr().out

    assert "12 played matches have a result upstream" in output
    assert "football-data.co.uk" in output


def test_a_clean_run_does_not_tell_you_to_go_and_fix_something(
    monkeypatch, league_dirs, capsys
):
    """The count is always shown; the advice only when the count is non-zero.

    A clean run should not print an instruction about a problem the reader does
    not have, which is the fastest way to make people stop reading a report.
    """
    def run_fixtures(league, season, dry_run=False):
        return {
            "added": 0,
            "rescheduled": 0,
            "needs_results": 0,
            "unknown_teams": [],
        }

    monkeypatch.setattr(refresh_data, "add_fixtures", run_fixtures)

    monkeypatch.setattr(
        refresh_data,
        "backfill_league_xg",
        lambda league, dry_run=False: {"xg_filled": 0, "rows": 0, "rows_unmatched": 0},
    )

    monkeypatch.setattr(refresh_data, "parse_arguments", lambda: argparse_namespace())

    refresh_data.main()

    output = capsys.readouterr().out

    assert "has a result upstream" not in output
    assert "Every played match on file has its result." in output


def test_a_failed_league_is_visible_in_the_table(monkeypatch, league_dirs, capsys):
    """A league that did not refresh must not look like one that had nothing
    to do, which is the failure this hides."""
    def run_fixtures(league, season, dry_run=False):
        if league == "LaLiga":
            raise UnderstatUnavailable("nothing listed")

        return {
            "added": 0,
            "rescheduled": 0,
            "needs_results": 0,
            "unknown_teams": [],
        }

    monkeypatch.setattr(refresh_data, "add_fixtures", run_fixtures)

    monkeypatch.setattr(
        refresh_data,
        "backfill_league_xg",
        lambda league, dry_run=False: {"xg_filled": 0, "rows": 0, "rows_unmatched": 0},
    )

    monkeypatch.setattr(refresh_data, "parse_arguments", lambda: argparse_namespace())

    refresh_data.main()

    output = capsys.readouterr().out

    assert "FAILED" in output
    assert "1 of 3 leagues could not be refreshed" in output

