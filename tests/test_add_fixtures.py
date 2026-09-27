"""
Tests for the pending-fixture script.

The script writes the rest of a season's schedule into the season CSVs, so the
thing worth testing is that it only ever adds what is missing: a played match
is never touched, a fixture already present is not duplicated, and a team the
CSV has never seen is reported rather than invented.
"""

import pandas as pd
import pytest

import add_fixtures


@pytest.fixture
def league_dir(tmp_path, monkeypatch):
    """
    Builds a two-season PremierLeague directory in a temporary location.

    The script resolves its paths relative to the working directory, so the
    tests chdir rather than patch the config, which keeps the real path logic
    under test.
    """
    directory = tmp_path / "football_data" / "PremierLeague"
    directory.mkdir(parents=True)

    monkeypatch.chdir(tmp_path)

    return directory


def write_season(directory, name, rows):
    """Writes a season CSV in the DD/MM/YYYY spelling the project uses."""
    path = directory / name

    pd.DataFrame(rows).to_csv(path, index=False)

    return path


def played_rows():
    """Two decided matches in the season under prediction."""
    return [
        {
            "Date": "12/08/2026",
            "Div": "E0",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Chelsea",
            "FTHG": 2,
            "FTAG": 1,
            "FTR": "H",
            "HS": 15,
            "AS": 8,
            "HST": 6,
            "AST": 3,
            "home_xg": 2.10,
            "away_xg": 0.90,
            "xg_source": "understat",
        },
        {
            "Date": "19/08/2026",
            "Div": "E0",
            "HomeTeam": "Chelsea",
            "AwayTeam": "Arsenal",
            "FTHG": 0,
            "FTAG": 0,
            "FTR": "D",
            "HS": 9,
            "AS": 11,
            "HST": 2,
            "AST": 4,
            "home_xg": 0.70,
            "away_xg": 1.10,
            "xg_source": "understat",
        },
    ]


def understat_fixtures(extra=()):
    """
    Builds an Understat schedule: the two played matches plus more.
    """
    rows = [
        {
            "date": pd.Timestamp("2026-08-12"),
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "home_goals": 2.0,
            "away_goals": 1.0,
            "home_xg": 2.1,
            "away_xg": 0.9,
            "played": True,
        },
        {
            "date": pd.Timestamp("2026-08-19"),
            "home_team": "Chelsea",
            "away_team": "Arsenal",
            "home_goals": 0.0,
            "away_goals": 0.0,
            "home_xg": 0.7,
            "away_xg": 1.1,
            "played": True,
        },
    ]

    rows += list(extra)

    return pd.DataFrame(rows)


def pending(date, home="Chelsea", away="Arsenal"):
    """A fixture that has not been played yet."""
    return {
        "date": pd.Timestamp(date),
        "home_team": home,
        "away_team": away,
        "home_goals": None,
        "away_goals": None,
        "home_xg": None,
        "away_xg": None,
        "played": False,
    }


def test_the_season_file_is_matched_exactly(league_dir, monkeypatch):
    """2026/2027 must not resolve to 2025-2026.csv.

    The file is named for the two years it spans, so a substring match on the
    start year alone would also find 2025-2026, since it contains 2026. The
    fixtures would then be added to the season before.
    """
    write_season(league_dir, "2025-2026.csv", played_rows())
    write_season(league_dir, "2026-2027.csv", played_rows())

    assert add_fixtures._season_file(
        "football_data/PremierLeague", "2026/2027"
    ).endswith("2026-2027.csv")

    assert add_fixtures._season_file(
        "football_data/PremierLeague", "2019/2020"
    ) is None


def test_pending_fixtures_are_added_without_a_result(league_dir, monkeypatch):
    """A fixture goes in with its date and sides, and nothing invented."""
    path = write_season(league_dir, "2026-2027.csv", played_rows())

    monkeypatch.setattr(
        add_fixtures,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures(
            [pending("2026-08-26"), pending("2026-09-02")]
        ),
    )

    add_fixtures.add_fixtures("PremierLeague", "2026/2027")

    result = pd.read_csv(path)

    assert len(result) == 4

    added = result[result["FTR"].isna()]

    assert len(added) == 2
    assert added["FTR"].isna().all()
    assert added["FTHG"].isna().all()
    assert added["FTAG"].isna().all()

    # A fixture that has not been played has no shots and no xG either, and a
    # zero there would be a measurement rather than a gap.
    assert added["HS"].isna().all()
    assert added["home_xg"].isna().all()

    assert added["Date"].tolist() == ["26/08/2026", "02/09/2026"]


def test_a_played_match_is_never_overwritten(league_dir, monkeypatch):
    """The result in the file wins, even where Understat carries one."""
    path = write_season(league_dir, "2026-2027.csv", played_rows())

    monkeypatch.setattr(
        add_fixtures,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures([pending("2026-08-26")]),
    )

    add_fixtures.add_fixtures("PremierLeague", "2026/2027")

    result = pd.read_csv(path)

    first = result[result["Date"] == "12/08/2026"].iloc[0]

    assert first["FTR"] == "H"
    assert first["FTHG"] == 2
    assert first["FTAG"] == 1


def test_running_twice_changes_nothing(league_dir, monkeypatch):
    """The script is a backfill, not an appender."""
    path = write_season(league_dir, "2026-2027.csv", played_rows())

    monkeypatch.setattr(
        add_fixtures,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures(
            [pending("2026-08-26"), pending("2026-09-02")]
        ),
    )

    add_fixtures.add_fixtures("PremierLeague", "2026/2027")
    first = path.read_text(encoding="utf-8")

    add_fixtures.add_fixtures("PremierLeague", "2026/2027")

    assert path.read_text(encoding="utf-8") == first


def test_the_column_order_and_shape_survive(league_dir, monkeypatch):
    """An added fixture is the same shape as the rows around it."""
    path = write_season(league_dir, "2026-2027.csv", played_rows())

    before = pd.read_csv(path).columns.tolist()

    monkeypatch.setattr(
        add_fixtures,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures([pending("2026-08-26")]),
    )

    add_fixtures.add_fixtures("PremierLeague", "2026/2027")

    assert pd.read_csv(path).columns.tolist() == before


def test_a_team_the_csv_has_never_seen_is_left_out(league_dir, monkeypatch):
    """An unfamiliar club is reported rather than written as a new one.

    Two spellings of the same club would split its Elo rating and its rolling
    form in two, and nothing downstream would notice.
    """
    path = write_season(league_dir, "2026-2027.csv", played_rows())

    monkeypatch.setattr(
        add_fixtures,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures(
            [pending("2026-08-26", "Wigan Athletic", "Arsenal")]
        ),
    )

    summary = add_fixtures.add_fixtures("PremierLeague", "2026/2027")

    assert summary["added"] == 0
    assert summary["unknown_teams"] == ["Wigan Athletic"]

    result = pd.read_csv(path)

    assert "Wigan Athletic" not in set(result["HomeTeam"])


def test_a_played_match_missing_from_the_csv_is_reported(
    league_dir, monkeypatch, capsys
):
    """A gap in the results is named, not filled in from a second source.

    Results belong to football-data.co.uk. A match that has been played but is
    not in the CSV is a gap in that source rather than a scheduling question, so
    it is listed for the user to resolve.
    """
    write_season(league_dir, "2026-2027.csv", played_rows()[:1])

    monkeypatch.setattr(
        add_fixtures,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures(),
    )

    summary = add_fixtures.add_fixtures("PremierLeague", "2026/2027")

    output = capsys.readouterr().out

    assert summary["needs_results"] == 1
    assert "2026-08-19" in output


def test_a_fixture_played_since_it_was_added_is_reported(
    league_dir, monkeypatch, capsys
):
    """A pending row that has since been played is the common case.

    A fixture is added while it is still upcoming, so the file holds a row with
    a blank result. Once the match is played the file is out of date, and until
    the results are downloaded that row reads as a match that has not happened:
    it is dropped from the features and from the Elo, and the club's recent form
    goes stale with nothing to say so. Counting only fixtures the file has never
    heard of misses this entirely, because the row is present and matches on
    date, home and away.
    """
    rows = played_rows()

    rows.append({
        "Date": "26/08/2026",
        "Div": "E0",
        "HomeTeam": "Chelsea",
        "AwayTeam": "Arsenal",
        "FTHG": None,
        "FTAG": None,
        "FTR": None,
        "HS": None,
        "AS": None,
        "HST": None,
        "AST": None,
        "home_xg": None,
        "away_xg": None,
        "xg_source": None,
    })

    write_season(league_dir, "2026-2027.csv", rows)

    # Understat now knows the score for a fixture the file added in advance.
    played_later = understat_fixtures()

    played_later.loc[len(played_later)] = {
        "date": pd.Timestamp("2026-08-26"),
        "home_team": "Chelsea",
        "away_team": "Arsenal",
        "home_goals": 2.0,
        "away_goals": 0.0,
        "home_xg": 1.8,
        "away_xg": 0.4,
        "played": True,
    }

    monkeypatch.setattr(
        add_fixtures,
        "get_league_fixtures",
        lambda slug, season: played_later,
    )

    summary = add_fixtures.add_fixtures("PremierLeague", "2026/2027")

    output = capsys.readouterr().out

    # The row is on file and matches, so it is not a fixture to add, but its
    # result is missing and that has to be said out loud.
    assert summary["added"] == 0
    assert summary["already_present"] == 3
    assert summary["needs_results"] == 1
    assert "2026-08-26" in output
    assert "football-data.co.uk" in output


def test_a_settled_fixture_is_not_reported_as_needing_results(
    league_dir, monkeypatch, capsys
):
    """A match the file already has the result for is not staleness."""
    write_season(league_dir, "2026-2027.csv", played_rows())

    monkeypatch.setattr(
        add_fixtures,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures(),
    )

    summary = add_fixtures.add_fixtures("PremierLeague", "2026/2027")

    assert summary["needs_results"] == 0
    # The count is always reported. The instruction to go and do something
    # about it is not, because a clean run should not tell you to go and fix
    # something that is not broken.
    assert "Refresh the season download" not in capsys.readouterr().out


def test_a_postponed_fixture_is_moved_rather_than_duplicated(
    league_dir, monkeypatch, capsys
):
    """A fixture that changes date has its row re-dated, not added again.

    Understat publishes the new date and the file still holds the old one, so
    the pair is on file at a date Understat no longer lists. Appending it would
    leave two rows for one match, and the old one with a blank result forever,
    which reads as a fixture that never happens.
    """
    rows = played_rows()

    rows.append({
        "Date": "26/08/2026",
        "Div": "E0",
        "HomeTeam": "Chelsea",
        "AwayTeam": "Arsenal",
        "FTHG": None,
        "FTAG": None,
        "FTR": None,
        "HS": None,
        "AS": None,
        "HST": None,
        "AST": None,
        "home_xg": None,
        "away_xg": None,
        "xg_source": None,
    })

    path = write_season(league_dir, "2026-2027.csv", rows)

    monkeypatch.setattr(
        add_fixtures,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures(
            [pending("2026-10-03")]
        ),
    )

    summary = add_fixtures.add_fixtures("PremierLeague", "2026/2027")

    result = pd.read_csv(path)

    assert summary["added"] == 0
    assert summary["rescheduled"] == 1

    # One row for the match, on the new date, still with no result.
    moved = result[
        (result["HomeTeam"] == "Chelsea") & (result["AwayTeam"] == "Arsenal")
        & (result["FTR"].isna())
    ]

    assert len(moved) == 1
    assert moved["Date"].tolist() == ["03/10/2026"]

    assert "26/08/2026" not in set(result["Date"])

    assert "moved date" in capsys.readouterr().out


def test_a_return_leg_is_added_rather_than_treated_as_a_reschedule(
    league_dir, monkeypatch
):
    """Meeting twice in a season is not a fixture that moved.

    This is the case that makes the reschedule pass dangerous if it is written
    carelessly. The first meeting has a result, so it is not a candidate for
    re-dating, and the second is a new row.
    """
    rows = played_rows()

    path = write_season(league_dir, "2026-2027.csv", rows)

    monkeypatch.setattr(
        add_fixtures,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures([pending("2026-11-07")]),
    )

    summary = add_fixtures.add_fixtures("PremierLeague", "2026/2027")

    result = pd.read_csv(path)

    assert summary["rescheduled"] == 0
    assert summary["added"] == 1
    assert len(result) == 3
    assert "07/11/2026" in set(result["Date"])


def test_a_played_row_is_never_re_dated(league_dir, monkeypatch):
    """A date a match was played on is a fact, not a fixture that moved."""
    path = write_season(league_dir, "2026-2027.csv", played_rows())

    monkeypatch.setattr(
        add_fixtures,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures(),
    )

    add_fixtures.add_fixtures("PremierLeague", "2026/2027")

    result = pd.read_csv(path)

    assert result["Date"].tolist() == ["12/08/2026", "19/08/2026"]


def test_a_dry_run_reports_reschedules_without_writing(
    league_dir, monkeypatch, capsys
):
    """The report can be had before anything touches the file."""
    rows = played_rows()

    rows.append({
        "Date": "26/08/2026",
        "Div": "E0",
        "HomeTeam": "Chelsea",
        "AwayTeam": "Arsenal",
        "FTHG": None,
        "FTAG": None,
        "FTR": None,
        "HS": None,
        "AS": None,
        "HST": None,
        "AST": None,
        "home_xg": None,
        "away_xg": None,
        "xg_source": None,
    })

    path = write_season(league_dir, "2026-2027.csv", rows)

    before = path.read_text(encoding="utf-8")

    monkeypatch.setattr(
        add_fixtures,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures([pending("2026-10-03")]),
    )

    summary = add_fixtures.add_fixtures(
        "PremierLeague", "2026/2027", dry_run=True
    )

    assert summary["rescheduled"] == 1
    assert path.read_text(encoding="utf-8") == before
    assert "moved date" in capsys.readouterr().out


def test_the_normalised_date_column_is_not_written_to_the_file(league_dir, monkeypatch):
    """Working state does not leak into the season file."""
    rows = played_rows()

    rows.append({
        "Date": "26/08/2026",
        "Div": "E0",
        "HomeTeam": "Chelsea",
        "AwayTeam": "Arsenal",
        "FTHG": None,
        "FTAG": None,
        "FTR": None,
    })

    path = write_season(league_dir, "2026-2027.csv", rows)

    before = pd.read_csv(path, nrows=0).columns.tolist()

    monkeypatch.setattr(
        add_fixtures,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures([pending("2026-10-03")]),
    )

    add_fixtures.add_fixtures("PremierLeague", "2026/2027")

    assert pd.read_csv(path, nrows=0).columns.tolist() == before


def test_a_dry_run_writes_nothing(league_dir, monkeypatch):
    """The report can be had before anything touches the file."""
    path = write_season(league_dir, "2026-2027.csv", played_rows())

    before = path.read_text(encoding="utf-8")

    monkeypatch.setattr(
        add_fixtures,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures([pending("2026-08-26")]),
    )

    summary = add_fixtures.add_fixtures(
        "PremierLeague", "2026/2027", dry_run=True
    )

    assert summary["added"] == 1
    assert path.read_text(encoding="utf-8") == before


def test_understat_team_names_are_translated(league_dir, monkeypatch):
    """The fixture is written in the spelling the CSV already uses.

    Understat lists a club by its registered name, which is not the name
    football-data.co.uk files it under. Written as it arrives, the fixture would
    be a club of its own with no rating and no form.
    """
    rows = played_rows()

    rows.append({
        "Date": "26/08/2026",
        "Div": "E0",
        "HomeTeam": "Man City",
        "AwayTeam": "Arsenal",
        "FTHG": 1,
        "FTAG": 1,
        "FTR": "D",
        "HS": 12,
        "AS": 10,
        "HST": 5,
        "AST": 3,
    })

    path = write_season(league_dir, "2026-2027.csv", rows)

    monkeypatch.setattr(
        add_fixtures,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures(
            [pending("2026-09-12", "Manchester City", "Arsenal")]
        ),
    )

    add_fixtures.add_fixtures("PremierLeague", "2026/2027")

    result = pd.read_csv(path)

    added = result[result["FTR"].isna()]

    assert added["HomeTeam"].tolist() == ["Man City"]
    assert "Manchester City" not in set(result["HomeTeam"])


def test_the_outcome_letter_is_only_given_where_there_is_a_score():
    """No score means no result, rather than a default."""
    assert add_fixtures._outcome(2, 1) == "H"
    assert add_fixtures._outcome(1, 1) == "D"
    assert add_fixtures._outcome(0, 2) == "A"
    assert add_fixtures._outcome(None, None) is None
    assert add_fixtures._outcome(1, None) is None
