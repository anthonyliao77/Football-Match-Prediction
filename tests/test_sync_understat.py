"""
Tests for the pending-fixture script.

The script writes the rest of a season's schedule into the season CSVs, so the
thing worth testing is that it only ever adds what is missing: a played match
is never touched, a fixture already present is not duplicated, and a team the
CSV has never seen is reported rather than invented.
"""

import pandas as pd
import pytest

import sync_understat


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
            "understat_id": "1",
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
            "understat_id": "2",
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
        "understat_id": "9",
    }


def stub_shots(monkeypatch, by_id=None, default=None):
    """
    Points the script at a fake per-match shot reader.

    A count is looked up by the match id, so one match can be made to fail while
    its neighbours succeed, which is the case worth testing.

    Parameters:
        monkeypatch: The pytest fixture.
        by_id (dict): Match id to a count dict, or to an exception to raise.
        default (dict): The counts for any id not named.
    """
    counts = default or {
        "home_shots": 12,
        "away_shots": 7,
        "home_on_target": 5,
        "away_on_target": 3,
    }

    by_id = by_id or {}

    def read(match_id):
        value = by_id.get(str(match_id), counts)

        if isinstance(value, Exception):
            raise value

        return value

    monkeypatch.setattr(sync_understat, "get_match_shots", read)


def test_the_season_file_is_matched_exactly(league_dir, monkeypatch):
    """2026/2027 must not resolve to 2025-2026.csv.

    The file is named for the two years it spans, so a substring match on the
    start year alone would also find 2025-2026, since it contains 2026. The
    fixtures would then be added to the season before.
    """
    write_season(league_dir, "2025-2026.csv", played_rows())
    write_season(league_dir, "2026-2027.csv", played_rows())

    assert sync_understat._season_file(
        "football_data/PremierLeague", "2026/2027"
    ).endswith("2026-2027.csv")

    assert sync_understat._season_file(
        "football_data/PremierLeague", "2019/2020"
    ) is None


def test_pending_fixtures_are_added_without_a_result(league_dir, monkeypatch):
    """A fixture goes in with its date and sides, and nothing invented."""
    path = write_season(league_dir, "2026-2027.csv", played_rows())

    monkeypatch.setattr(
        sync_understat,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures(
            [pending("2026-08-26"), pending("2026-09-02")]
        ),
    )

    sync_understat.sync_league("PremierLeague", "2026/2027")

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
        sync_understat,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures([pending("2026-08-26")]),
    )

    sync_understat.sync_league("PremierLeague", "2026/2027")

    result = pd.read_csv(path)

    first = result[result["Date"] == "12/08/2026"].iloc[0]

    assert first["FTR"] == "H"
    assert first["FTHG"] == 2
    assert first["FTAG"] == 1


def test_running_twice_changes_nothing(league_dir, monkeypatch):
    """The script is a backfill, not an appender."""
    path = write_season(league_dir, "2026-2027.csv", played_rows())

    monkeypatch.setattr(
        sync_understat,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures(
            [pending("2026-08-26"), pending("2026-09-02")]
        ),
    )

    sync_understat.sync_league("PremierLeague", "2026/2027")
    first = path.read_text(encoding="utf-8")

    sync_understat.sync_league("PremierLeague", "2026/2027")

    assert path.read_text(encoding="utf-8") == first


def test_the_column_order_and_shape_survive(league_dir, monkeypatch):
    """An added fixture is the same shape as the rows around it."""
    path = write_season(league_dir, "2026-2027.csv", played_rows())

    before = pd.read_csv(path).columns.tolist()

    monkeypatch.setattr(
        sync_understat,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures([pending("2026-08-26")]),
    )

    sync_understat.sync_league("PremierLeague", "2026/2027")

    assert pd.read_csv(path).columns.tolist() == before


def test_a_team_the_csv_has_never_seen_is_left_out(league_dir, monkeypatch):
    """An unfamiliar club is reported rather than written as a new one.

    Two spellings of the same club would split its Elo rating and its rolling
    form in two, and nothing downstream would notice.
    """
    path = write_season(league_dir, "2026-2027.csv", played_rows())

    monkeypatch.setattr(
        sync_understat,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures(
            [pending("2026-08-26", "Wigan Athletic", "Arsenal")]
        ),
    )

    summary = sync_understat.sync_league("PremierLeague", "2026/2027")

    assert summary["added"] == 0
    assert summary["unknown_teams"] == ["Wigan Athletic"]

    result = pd.read_csv(path)

    assert "Wigan Athletic" not in set(result["HomeTeam"])


def test_a_played_match_missing_from_the_csv_is_added_with_its_result(
    league_dir, monkeypatch
):
    """A match the season file has never heard of is added, decided.

    It used to be reported and left out, on the grounds that results belong to
    football-data.co.uk. That was true and it left the season permanently short
    a match that had been played, which is the one thing the file is for.
    """
    path = write_season(league_dir, "2026-2027.csv", played_rows()[:1])

    monkeypatch.setattr(
        sync_understat,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures(),
    )
    stub_shots(monkeypatch)

    summary = sync_understat.sync_league("PremierLeague", "2026/2027")

    result = pd.read_csv(path)

    assert summary["results_added"] == 1
    assert summary["needs_results"] == 0
    assert len(result) == 2

    added = result[result["Date"] == "19/08/2026"].iloc[0]

    assert added["FTHG"] == 0
    assert added["FTAG"] == 0
    assert added["FTR"] == "D"
    # The 19/08 goalless draw had shots in it, which is what separates it from
    # a fixture that was never played at all.
    assert added["HS"] == 12
    assert added["AST"] == 3


def test_a_filled_result_is_not_filled_again(league_dir, monkeypatch):
    """The second run has nothing to do and spends no requests.

    Results are fetched per match, so a script that re-read the shots for a
    match it had already written would cost a request per match per run, every
    run, for data it already had."""
    path = write_season(league_dir, "2026-2027.csv", played_rows()[:1])

    monkeypatch.setattr(
        sync_understat,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures(),
    )

    spent = []

    def read(match_id):
        spent.append(match_id)

        return {
            "home_shots": 12,
            "away_shots": 7,
            "home_on_target": 5,
            "away_on_target": 3,
        }

    monkeypatch.setattr(sync_understat, "get_match_shots", read)

    sync_understat.sync_league("PremierLeague", "2026/2027")

    first = pd.read_csv(path)
    spent.clear()

    summary = sync_understat.sync_league("PremierLeague", "2026/2027")

    second = pd.read_csv(path)

    assert spent == []
    assert summary["results_filled"] == 0
    assert summary["results_added"] == 0
    assert summary["match_requests"] == 0
    assert first.equals(second)


def test_a_fixture_played_since_it_was_added_is_filled_in(
    league_dir, monkeypatch, capsys
):
    """A pending row that has since been played is the common case, and it is
    now the case the script exists to fix.

    A fixture is added while it is still upcoming, so the file holds a row with
    a blank result. Once the match is played the row is out of date, and left
    blank it reads as a match that has not happened: it is dropped from the
    features and from the Elo, and the club's recent form goes stale with nothing
    to say so. Counting only fixtures the file has never heard of missed this
    entirely, because the row is present and matches on date, home and away.
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
        "understat_id": "3",
    }

    monkeypatch.setattr(
        sync_understat,
        "get_league_fixtures",
        lambda slug, season: played_later,
    )
    stub_shots(monkeypatch, by_id={"3": {
        "home_shots": 18,
        "away_shots": 6,
        "home_on_target": 7,
        "away_on_target": 2,
    }})

    summary = sync_understat.sync_league("PremierLeague", "2026/2027")

    result = pd.read_csv(league_dir / "2026-2027.csv")

    # The row was on file and matched, so it is not a fixture to add. It is
    # filled in where it stands, which is the point.
    assert summary["added"] == 0
    assert summary["already_present"] == 3
    assert summary["results_filled"] == 1
    assert summary["needs_results"] == 0
    assert len(result) == 3

    filled = result[result["Date"] == "26/08/2026"].iloc[0]

    assert filled["FTHG"] == 2
    assert filled["FTAG"] == 0
    assert filled["FTR"] == "H"
    assert filled["HS"] == 18
    assert filled["AST"] == 2


def test_a_settled_fixture_is_not_reported_as_needing_results(
    league_dir, monkeypatch, capsys
):
    """A match the file already has the result for is not staleness."""
    write_season(league_dir, "2026-2027.csv", played_rows())

    monkeypatch.setattr(
        sync_understat,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures(),
    )

    summary = sync_understat.sync_league("PremierLeague", "2026/2027")

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
        sync_understat,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures(
            [pending("2026-10-03")]
        ),
    )

    summary = sync_understat.sync_league("PremierLeague", "2026/2027")

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
        sync_understat,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures([pending("2026-11-07")]),
    )

    summary = sync_understat.sync_league("PremierLeague", "2026/2027")

    result = pd.read_csv(path)

    assert summary["rescheduled"] == 0
    assert summary["added"] == 1
    assert len(result) == 3
    assert "07/11/2026" in set(result["Date"])


def test_a_played_row_is_never_re_dated(league_dir, monkeypatch):
    """A date a match was played on is a fact, not a fixture that moved."""
    path = write_season(league_dir, "2026-2027.csv", played_rows())

    monkeypatch.setattr(
        sync_understat,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures(),
    )

    sync_understat.sync_league("PremierLeague", "2026/2027")

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
        sync_understat,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures([pending("2026-10-03")]),
    )

    summary = sync_understat.sync_league(
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
        sync_understat,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures([pending("2026-10-03")]),
    )

    sync_understat.sync_league("PremierLeague", "2026/2027")

    assert pd.read_csv(path, nrows=0).columns.tolist() == before


def test_a_dry_run_writes_nothing(league_dir, monkeypatch):
    """The report can be had before anything touches the file."""
    path = write_season(league_dir, "2026-2027.csv", played_rows())

    before = path.read_text(encoding="utf-8")

    monkeypatch.setattr(
        sync_understat,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures([pending("2026-08-26")]),
    )

    summary = sync_understat.sync_league(
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
        sync_understat,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures(
            [pending("2026-09-12", "Manchester City", "Arsenal")]
        ),
    )

    sync_understat.sync_league("PremierLeague", "2026/2027")

    result = pd.read_csv(path)

    added = result[result["FTR"].isna()]

    assert added["HomeTeam"].tolist() == ["Man City"]
    assert "Manchester City" not in set(result["HomeTeam"])


def test_the_outcome_letter_is_only_given_where_there_is_a_score():
    """No score means no result, rather than a default."""
    assert sync_understat._outcome(2, 1) == "H"
    assert sync_understat._outcome(1, 1) == "D"
    assert sync_understat._outcome(0, 2) == "A"
    assert sync_understat._outcome(None, None) is None
    assert sync_understat._outcome(1, None) is None


def blank_row(date="26/08/2026", home="Chelsea", away="Arsenal"):
    """A fixture the file added before kickoff and is still holding open."""
    return {
        "Date": date,
        "Div": "E0",
        "HomeTeam": home,
        "AwayTeam": away,
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
    }


def played_on(date, home, away, home_goals, away_goals, match_id="3"):
    """An Understat fixture that has been played."""
    return {
        "date": pd.Timestamp(date),
        "home_team": home,
        "away_team": away,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "home_xg": 1.4,
        "away_xg": 0.7,
        "played": True,
        "understat_id": match_id,
    }


def test_a_score_is_never_written_without_its_shot_counts(league_dir, monkeypatch):
    """The rule the whole design turns on.

    src/features.py coerces a missing number to zero on purpose, so a row given
    a score and no shots is not a row with blanks in it. It is a row claiming
    the club managed no shots, and nothing anywhere would say so: the features
    would sum, the model would predict, and the answer would be quietly wrong by
    about twenty shots. So a match whose shot counts cannot be read is left
    blank, score and all."""
    path = write_season(
        league_dir, "2026-2027.csv", [*played_rows(), blank_row()]
    )

    monkeypatch.setattr(
        sync_understat,
        "get_league_fixtures",
        lambda slug, season: pd.concat(
            [understat_fixtures(), pd.DataFrame([
                played_on("2026-08-26", "Chelsea", "Arsenal", 2.0, 0.0)
            ])],
            ignore_index=True,
        ),
    )
    stub_shots(monkeypatch, by_id={
        "3": sync_understat.UnderstatUnavailable("HTTP 500")
    })

    summary = sync_understat.sync_league("PremierLeague", "2026/2027")

    result = pd.read_csv(path)

    row = result[result["Date"] == "26/08/2026"].iloc[0]

    assert summary["results_filled"] == 0
    assert summary["needs_results"] == 1

    # Every cell still blank. The score was in hand and was still not written.
    for column in ("FTHG", "FTAG", "FTR", "HS", "AS", "HST", "AST"):
        assert pd.isna(row[column]), column


def test_a_goalless_draw_with_no_shots_is_not_written(league_dir, monkeypatch, capsys):
    """A 0-0 where neither side had a shot is not a football match.

    The leagues do record abandoned and awarded games, and writing one as a
    played 0-0 would put a match in the season that never happened, with a form
    entry and an Elo result attached to it for both clubs."""
    path = write_season(
        league_dir, "2026-2027.csv", [*played_rows(), blank_row()]
    )

    monkeypatch.setattr(
        sync_understat,
        "get_league_fixtures",
        lambda slug, season: pd.concat(
            [understat_fixtures(), pd.DataFrame([
                played_on("2026-08-26", "Chelsea", "Arsenal", 0.0, 0.0)
            ])],
            ignore_index=True,
        ),
    )
    stub_shots(monkeypatch, by_id={"3": {
        "home_shots": 0,
        "away_shots": 0,
        "home_on_target": 0,
        "away_on_target": 0,
    }})

    summary = sync_understat.sync_league("PremierLeague", "2026/2027")

    result = pd.read_csv(path)
    output = capsys.readouterr().out

    row = result[result["Date"] == "26/08/2026"].iloc[0]

    assert summary["results_filled"] == 0
    assert summary["needs_results"] == 1
    assert pd.isna(row["FTR"])
    assert "probably not a match that was played" in output


def test_a_real_goalless_draw_is_still_written(league_dir, monkeypatch):
    """The abandoned-match guard has to be narrow enough to miss a real 0-0.

    0-0 with shots in it is an ordinary result and the commonest one in the
    league. Skipping those too would quietly stop the pipeline recording any of
    them."""
    path = write_season(
        league_dir, "2026-2027.csv", [*played_rows(), blank_row()]
    )

    monkeypatch.setattr(
        sync_understat,
        "get_league_fixtures",
        lambda slug, season: pd.concat(
            [understat_fixtures(), pd.DataFrame([
                played_on("2026-08-26", "Chelsea", "Arsenal", 0.0, 0.0)
            ])],
            ignore_index=True,
        ),
    )
    stub_shots(monkeypatch, by_id={"3": {
        "home_shots": 14,
        "away_shots": 9,
        "home_on_target": 0,
        "away_on_target": 0,
    }})

    summary = sync_understat.sync_league("PremierLeague", "2026/2027")

    result = pd.read_csv(path)
    row = result[result["Date"] == "26/08/2026"].iloc[0]

    assert summary["results_filled"] == 1
    assert row["FTR"] == "D"
    assert row["FTHG"] == 0
    assert row["HS"] == 14


def test_an_existing_result_is_never_overwritten(league_dir, monkeypatch):
    """football-data.co.uk stays the authority on every row that has a result.

    Two spellings of one match would also mean the season held the match twice,
    and the second copy would be trained on as though it were a real game."""
    rows = played_rows()

    # The file says Arsenal won 2-1. Understat says it was a draw.
    lying = understat_fixtures()
    lying.loc[0, ["home_goals", "away_goals"]] = [1.0, 1.0]

    path = write_season(league_dir, "2026-2027.csv", rows)

    monkeypatch.setattr(
        sync_understat,
        "get_league_fixtures",
        lambda slug, season: lying,
    )
    stub_shots(monkeypatch)

    summary = sync_understat.sync_league("PremierLeague", "2026/2027")

    result = pd.read_csv(path)
    row = result[result["Date"] == "12/08/2026"].iloc[0]

    assert row["FTHG"] == 2
    assert row["FTAG"] == 1
    assert row["FTR"] == "H"
    assert summary["results_filled"] == 0
    assert summary["disagreements"] == 1


def test_a_disagreement_is_named_rather_than_resolved(league_dir, monkeypatch, capsys):
    """Silently picking a winner would be a rewrite with no way back.

    The two sources disagreeing is worth seeing. Which of them is right is not
    something a script can decide, and it is the one case where filling in the
    blanks would be actively harmful."""
    lying = understat_fixtures()
    lying.loc[0, ["home_goals", "away_goals"]] = [1.0, 1.0]

    write_season(league_dir, "2026-2027.csv", played_rows())

    monkeypatch.setattr(
        sync_understat,
        "get_league_fixtures",
        lambda slug, season: lying,
    )
    stub_shots(monkeypatch)

    sync_understat.sync_league("PremierLeague", "2026/2027")

    output = capsys.readouterr().out

    assert "sources disagree" in output
    assert "Arsenal v Chelsea" in output
    assert "file H vs Understat D" in output


def test_a_result_records_that_understat_wrote_it(league_dir, monkeypatch):
    """Provenance, so a fetched value is told apart from a downloaded one.

    Only the rows this script writes are marked. Nothing writes the other value
    on purpose: a column that says football-data.co.uk on every historical row
    would be a column nobody could trust to mean anything."""
    path = write_season(
        league_dir,
        "2026-2027.csv",
        [
            {**row, "result_source": "football-data"}
            if row["FTR"] else row
            for row in [*played_rows(), blank_row()]
        ],
    )

    monkeypatch.setattr(
        sync_understat,
        "get_league_fixtures",
        lambda slug, season: pd.concat(
            [understat_fixtures(), pd.DataFrame([
                played_on("2026-08-26", "Chelsea", "Arsenal", 2.0, 0.0)
            ])],
            ignore_index=True,
        ),
    )
    stub_shots(monkeypatch)

    sync_understat.sync_league("PremierLeague", "2026/2027")

    result = pd.read_csv(path)

    filled = result[result["Date"] == "26/08/2026"].iloc[0]
    downloaded = result[result["Date"] == "12/08/2026"].iloc[0]

    assert filled["result_source"] == "understat"
    assert downloaded["result_source"] == "football-data"


def test_nothing_is_written_when_the_file_lacks_a_shots_column(league_dir, monkeypatch, capsys):
    """A file that cannot hold a whole match gets no half of one.

    The guard that stops a score being written without its shot counts has to
    start with the file not having a shots column to write them into."""
    rows = [
        {column: value for column, value in row.items() if column != "HST"}
        for row in [*played_rows(), blank_row()]
    ]

    path = write_season(league_dir, "2026-2027.csv", rows)

    monkeypatch.setattr(
        sync_understat,
        "get_league_fixtures",
        lambda slug, season: pd.concat(
            [understat_fixtures(), pd.DataFrame([
                played_on("2026-08-26", "Chelsea", "Arsenal", 2.0, 0.0)
            ])],
            ignore_index=True,
        ),
    )

    spent = []
    monkeypatch.setattr(
        sync_understat,
        "get_match_shots",
        lambda match_id: spent.append(match_id),
    )

    summary = sync_understat.sync_league("PremierLeague", "2026/2027")

    output = capsys.readouterr().out
    result = pd.read_csv(path)
    row = result[result["Date"] == "26/08/2026"].iloc[0]

    assert summary["results_filled"] == 0
    assert spent == []
    assert "has no HST" in output
    assert "half-complete match" in output
    assert pd.isna(row["FTR"])
    assert "HST" not in result.columns


def test_a_dry_run_writes_no_result(league_dir, monkeypatch):
    """--dry-run has to cover the fill, or it reports a change it then makes."""
    path = write_season(
        league_dir, "2026-2027.csv", [*played_rows(), blank_row()]
    )

    monkeypatch.setattr(
        sync_understat,
        "get_league_fixtures",
        lambda slug, season: pd.concat(
            [understat_fixtures(), pd.DataFrame([
                played_on("2026-08-26", "Chelsea", "Arsenal", 2.0, 0.0)
            ])],
            ignore_index=True,
        ),
    )
    stub_shots(monkeypatch)

    summary = sync_understat.sync_league(
        "PremierLeague", "2026/2027", dry_run=True
    )

    result = pd.read_csv(path)
    row = result[result["Date"] == "26/08/2026"].iloc[0]

    # The plan is still reported, so a dry run says what a real run would do.
    assert summary["results_filled"] == 1
    assert summary["match_requests"] == 1
    assert pd.isna(row["FTR"])


def test_a_row_with_a_result_but_no_shots_is_reported_not_filled(
    league_dir, monkeypatch, capsys
):
    """A half-complete row that arrived from somewhere else is not left silent.

    The rule that a result is never written without its shot counts protects the
    rows this script writes. A row that already has a result and no shots is the
    same corrupt shape arriving by another route, and the features read the blank
    as zero either way. It is not filled, because the goals on it came from
    football-data.co.uk and the shots would come from Understat, and mixing the
    two inside one match is worse than an honest gap. It is named instead."""
    rows = played_rows()

    # A result with no shot counts, as a hand-edited file or an older script
    # would leave it. The 19/08 is Understat's own second fixture, so this is a
    # row the refresh looks at and then declines to touch.
    rows[1].update({"HS": None, "AS": None, "HST": None, "AST": None})

    path = write_season(league_dir, "2026-2027.csv", rows)

    monkeypatch.setattr(
        sync_understat,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures(),
    )
    stub_shots(monkeypatch)

    summary = sync_understat.sync_league("PremierLeague", "2026/2027")

    output = capsys.readouterr().out
    result = pd.read_csv(path)
    row = result[result["Date"] == "19/08/2026"].iloc[0]

    assert summary["partial_results"] == 1
    assert "no HS, AS, HST, AST" in output
    assert "Chelsea v Arsenal" in output

    # Untouched. The goals were not Understat's to confirm and the shots were
    # not written beside them.
    assert row["FTHG"] == 0
    assert row["FTAG"] == 0
    assert row["FTR"] == "D"
    assert pd.isna(row["HS"])


def test_a_whole_row_is_not_reported_as_partial(league_dir, monkeypatch):
    """The report is only worth reading if it stays empty in the normal case."""
    write_season(league_dir, "2026-2027.csv", played_rows())

    monkeypatch.setattr(
        sync_understat,
        "get_league_fixtures",
        lambda slug, season: understat_fixtures(),
    )
    stub_shots(monkeypatch)

    summary = sync_understat.sync_league("PremierLeague", "2026/2027")

    assert summary["partial_results"] == 0
