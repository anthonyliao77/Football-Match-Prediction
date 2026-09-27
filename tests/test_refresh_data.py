"""
Tests for the refresh script.

The script itself holds no fetching logic, so what is worth testing is the
behaviour that comes from orchestrating: that both steps are attempted, that one
league failing does not cost the others, that the exit status says so, and that
the number it leads with is the one a person running it needs, which is how far
behind their results download is.
"""

import sys

import pandas as pd
import pytest

import refresh_data
from backfill_xg import UnderstatDataError
from config import LEAGUES
from sync_understat import UnderstatUnavailable


def argparse_namespace(league="all", dry_run=False, require_fresh=False):
    """Builds a stand-in for the parsed command line."""
    import argparse

    return argparse.Namespace(
        league=league, season=None, dry_run=dry_run, require_fresh=require_fresh
    )


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

    def run_fixtures(league, season, dry_run=False, **kwargs):
        calls["fixtures"].append((league, season, dry_run))

        if fixtures_error is not None:
            raise fixtures_error

        if fixtures is not None:
            return fixtures

        return fixtures_summary(
            added=3,
            already_present=10,
            rescheduled=1,
            needs_results=4,
            results_filled=2,
            results_added=1,
            match_requests=3,
        )

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

    monkeypatch.setattr(refresh_data, "sync_league", run_fixtures)
    monkeypatch.setattr(refresh_data, "backfill_league_xg", run_xg)

    return calls


# One place that knows what sync_league reports. Each test then states only the
# counts it is about, so adding a count to the real contract does not mean
# editing eleven stubs, and a test cannot pass against a shape the real step no
# longer returns.
def fixtures_summary(**counts):
    summary = {
        "added": 0,
        "already_present": 0,
        "rescheduled": 0,
        "needs_results": 0,
        "results_filled": 0,
        "results_added": 0,
        "disagreements": 0,
        "match_requests": 0,
        "unknown_teams": [],
    }
    summary.update(counts)

    return summary


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
        "sync_league",
        lambda league, season, dry_run=False: order.append("fixtures")
        or fixtures_summary(),
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

        return fixtures_summary(
            added=1,
            rescheduled=0,
            needs_results=0,
        )

    monkeypatch.setattr(refresh_data, "sync_league", run_fixtures)

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

        return fixtures_summary(
            added=0,
            rescheduled=0,
            needs_results=0,
        )

    monkeypatch.setattr(refresh_data, "sync_league", run_fixtures)

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


def test_the_count_of_unwritten_matches_is_the_headline(monkeypatch, league_dirs, capsys):
    """The number the person running it needs is how many matches have no
    result written.

    Neither underlying script can report this. A fixture added before kickoff
    sits in the file with a blank result, matches on date and sides, and is
    counted as already present by both of them, so nothing anywhere says a
    played match is still unwritten. The refresh exists largely to say it.
    """
    stub(monkeypatch)

    monkeypatch.setattr(refresh_data, "parse_arguments", lambda: argparse_namespace())

    refresh_data.main()

    output = capsys.readouterr().out

    assert "12 played matches are still without a result" in output


def test_a_clean_run_does_not_tell_you_to_go_and_fix_something(
    monkeypatch, league_dirs, capsys
):
    """The count is always shown; the advice only when the count is non-zero.

    A clean run should not print an instruction about a problem the reader does
    not have, which is the fastest way to make people stop reading a report.
    """
    def run_fixtures(league, season, dry_run=False):
        return fixtures_summary(
            added=0,
            rescheduled=0,
            needs_results=0,
        )

    monkeypatch.setattr(refresh_data, "sync_league", run_fixtures)

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

        return fixtures_summary(
            added=0,
            rescheduled=0,
            needs_results=0,
        )

    monkeypatch.setattr(refresh_data, "sync_league", run_fixtures)

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


def test_a_failed_league_does_not_produce_an_all_clear(monkeypatch, league_dirs, capsys):
    """A league that was never checked cannot be reported as clean.

    This was the worst defect in the script: the stale count skipped leagues
    that had failed, so a league with 99 stale matches and a league with none
    both summed to zero, and the run ended by saying every played match on
    file has its result. The count was the entire point of the command.
    """
    def run_fixtures(league, season, dry_run=False):
        if league == "PremierLeague":
            raise UnderstatUnavailable("nothing listed")

        return fixtures_summary(
            added=0,
            rescheduled=0,
            needs_results=0,
        )

    monkeypatch.setattr(refresh_data, "sync_league", run_fixtures)

    monkeypatch.setattr(
        refresh_data,
        "backfill_league_xg",
        lambda league, dry_run=False: {
            "xg_filled": 0,
            "rows": 0,
            "rows_unmatched": 0,
        },
    )

    monkeypatch.setattr(refresh_data, "parse_arguments", lambda: argparse_namespace())

    refresh_data.main()

    output = capsys.readouterr().out

    assert "Every played match on file has its result" not in output
    assert "count is unknown" in output


def test_a_league_that_was_not_checked_shows_a_dash_not_a_zero(monkeypatch, league_dirs, capsys):
    """A dash and a zero look identical in a table and mean opposite things."""
    def run_fixtures(league, season, dry_run=False):
        if league == "PremierLeague":
            raise UnderstatUnavailable("nothing listed")

        return fixtures_summary(
            added=0,
            rescheduled=0,
            needs_results=0,
        )

    monkeypatch.setattr(refresh_data, "sync_league", run_fixtures)

    monkeypatch.setattr(
        refresh_data,
        "backfill_league_xg",
        lambda league, dry_run=False: {
            "xg_filled": 0,
            "rows": 0,
            "rows_unmatched": 0,
        },
    )

    monkeypatch.setattr(refresh_data, "parse_arguments", lambda: argparse_namespace())

    refresh_data.main()

    output = capsys.readouterr().out

    failed_line = next(
        line
        for line in output.splitlines()
        if "PremierLeague" in line and "FAILED" in line
    )

    # The dash is the Stale column, so the last cell before the error text.
    # Asserting on the last word of the line would be asserting on the error.
    cells = failed_line.split("FAILED")[0].split()

    assert cells[-1] == "-"
    assert "A dash is not a zero" in output


def test_a_partly_failed_league_still_shows_what_it_wrote(monkeypatch, league_dirs, capsys):
    """Fixtures written and then an xG failure is not a league where nothing
    happened. The counts are on disk already, and hiding them behind a bare
    FAILED leaves the reader unable to tell a rerun is safe from a rerun that
    would be the first thing to write anything."""
    stub(monkeypatch)

    def run_xg(league, dry_run=False):
        raise UnderstatDataError("no 2026/2027")

    monkeypatch.setattr(refresh_data, "backfill_league_xg", run_xg)

    monkeypatch.setattr(refresh_data, "parse_arguments", lambda: argparse_namespace())

    refresh_data.main()

    output = capsys.readouterr().out

    line = next(
        line
        for line in output.splitlines()
        if "PremierLeague" in line and "FAILED" in line
    )

    # Three fixtures added and one moved, per the stub, shown on the failed line.
    assert "3" in line.split("FAILED")[0]
    assert "1" in line.split("FAILED")[0]
    assert "xG:" in line


def test_require_fresh_exits_non_zero_when_results_are_missing(
    monkeypatch, league_dirs
):
    """The stale download is the failure a scheduled run is most likely to
    miss, because the refresh itself succeeds. Both halves report clean, the
    exit status is zero, and the model trains on last month."""
    stub(monkeypatch)

    monkeypatch.setattr(
        refresh_data,
        "parse_arguments",
        lambda: argparse_namespace(require_fresh=True),
    )

    assert refresh_data.main() == 1


def test_require_fresh_exits_zero_when_there_is_nothing_to_fetch(
    monkeypatch, league_dirs
):
    """The flag has to be able to pass, or a schedule wired to it disables
    itself after the first run by alerting on every success."""
    def run_fixtures(league, season, dry_run=False):
        return fixtures_summary(
            added=3,
            rescheduled=1,
            needs_results=0,
        )

    monkeypatch.setattr(refresh_data, "sync_league", run_fixtures)

    monkeypatch.setattr(
        refresh_data,
        "backfill_league_xg",
        lambda league, dry_run=False: {
            "xg_filled": 12,
            "rows": 700,
            "rows_unmatched": 0,
        },
    )

    monkeypatch.setattr(
        refresh_data,
        "parse_arguments",
        lambda: argparse_namespace(require_fresh=True),
    )

    assert refresh_data.main() == 0


def test_require_fresh_also_fails_on_an_unchecked_league(monkeypatch, league_dirs):
    """An unknown stale count is not a passing one. Requiring fresh has to
    refuse to certify a league it could not look at, or it is a worse guard
    than not asking."""
    def run_fixtures(league, season, dry_run=False):
        if league == "PremierLeague":
            raise UnderstatUnavailable("nothing listed")

        return fixtures_summary(
            added=0,
            rescheduled=0,
            needs_results=0,
        )

    monkeypatch.setattr(refresh_data, "sync_league", run_fixtures)

    monkeypatch.setattr(
        refresh_data,
        "backfill_league_xg",
        lambda league, dry_run=False: {
            "xg_filled": 0,
            "rows": 0,
            "rows_unmatched": 0,
        },
    )

    monkeypatch.setattr(
        refresh_data,
        "parse_arguments",
        lambda: argparse_namespace(require_fresh=True),
    )

    assert refresh_data.main() == 1

def test_the_require_fresh_flag_is_parsed_by_the_command_line(monkeypatch, league_dirs):
    """The other tests replace the parsed arguments wholesale, which means a
    flag that was never added to the parser would pass all of them. This one
    goes through the real parser."""
    def run_fixtures(league, season, dry_run=False):
        return fixtures_summary(
            added=0,
            rescheduled=0,
            needs_results=7,
        )

    monkeypatch.setattr(refresh_data, "sync_league", run_fixtures)

    monkeypatch.setattr(
        refresh_data,
        "backfill_league_xg",
        lambda league, dry_run=False: {
            "xg_filled": 0,
            "rows": 0,
            "rows_unmatched": 0,
        },
    )

    monkeypatch.setattr(
        sys, "argv", ["refresh_data.py", "--require-fresh", "--dry-run"]
    )

    assert refresh_data.main() == 1


def test_results_filled_and_added_are_reported_as_one_column(
    monkeypatch, league_dirs, capsys
):
    """Filling a blank row and adding a missing one are both a result arriving,
    and the reader wants the total rather than two numbers to add up."""
    stub(monkeypatch, fixtures=fixtures_summary(
        needs_results=2,
        results_filled=2,
        results_added=1,
        match_requests=3,
    ))

    monkeypatch.setattr(refresh_data, "parse_arguments", lambda: argparse_namespace())

    refresh_data.main()

    output = capsys.readouterr().out

    # The stubbed summary is applied to all three leagues, so the total is
    # three times one run's counts: 9 results arrived, 6 matches are still
    # unwritten. The two are unrelated numbers. One is progress and the other
    # is what is left, and the old report had only the second of them.
    assert "Results filled from Understat" in output
    assert "6 played matches are still without a result" in output


def test_a_disagreement_is_surfaced_in_the_table(monkeypatch, league_dirs, capsys):
    """The fixtures step names the match, but only in its own output.

    A refresh is often run where the detail has scrolled past, and a source
    disagreement is the one thing in that output a human has to look at."""
    stub(monkeypatch, fixtures=fixtures_summary(
        disagreements=2,
        needs_results=0,
    ))

    monkeypatch.setattr(refresh_data, "parse_arguments", lambda: argparse_namespace())

    refresh_data.main()

    output = capsys.readouterr().out

    assert "2 source disagreement(s)" in output


def test_the_column_no_longer_calls_a_result_gap_stale(monkeypatch, league_dirs, capsys):
    """The old heading was advice-free but the legend under it was not.

    It read as a thing to go and fix by downloading, which is exactly the
    reading the new behaviour removed."""
    stub(monkeypatch, fixtures=fixtures_summary(needs_results=0))

    monkeypatch.setattr(refresh_data, "parse_arguments", lambda: argparse_namespace())

    refresh_data.main()

    output = capsys.readouterr().out

    assert "Unwritten" in output
    assert "Every played match on file has its result" in output
