"""
Tests for the Understat xG backfill script.
"""

import numpy as np
import pandas as pd
import pytest

import backfill_xg
from config import TEAM_NAME_MAP
from src.understat_loader import UnderstatDataError
from src.xg import SOURCE_COLUMN, XG_COLUMNS


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
    """
    Writes a season CSV in the DD/MM/YYYY spelling football-data.co.uk uses.
    """
    path = directory / name

    pd.DataFrame(rows).to_csv(path, index=False)

    return path


def understat_rows(rows):
    """
    Builds an Understat schedule holding xG.
    """
    return pd.DataFrame(rows)


def base_rows():
    """Two matches, one already carrying xG and one bare."""
    return [
        {
            "Date": "12/08/2023",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Chelsea",
            "HS": 15,
            "AS": 8,
            "HST": 6,
            "AST": 3,
        },
        {
            "Date": "13/08/2023",
            "HomeTeam": "Liverpool",
            "AwayTeam": "Everton",
            "HS": 20,
            "AS": 6,
            "HST": 7,
            "AST": 2,
        },
    ]


def matching_understat():
    """Understat data for the two matches in base_rows, with one unmatched."""
    return understat_rows([
        {
            "date": "2023-08-12",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "home_xg": 2.10,
            "away_xg": 0.90,
        },
        {
            "date": "2023-08-13",
            "home_team": "Liverpool",
            "away_team": "Everton",
            "home_xg": 2.80,
            "away_xg": 0.70,
        },
        {
            "date": "2023-08-14",
            "home_team": "Fulham",
            "away_team": "Brentford",
            "home_xg": 1.10,
            "away_xg": 1.30,
        },
    ])


def test_backfill_fills_the_gaps_it_is_given(league_dir, monkeypatch):
    """A bare CSV comes out with xG, and the dates are still readable."""
    path = write_season(league_dir, "E0.csv", base_rows())

    monkeypatch.setattr(
        backfill_xg, "load_understat_data", lambda **kwargs: matching_understat()
    )

    backfill_xg.backfill_league_xg("PremierLeague")

    result = pd.read_csv(path)

    assert result["home_xg"].tolist() == [2.10, 2.80]
    assert result["away_xg"].tolist() == [0.90, 0.70]
    assert (result[SOURCE_COLUMN] == "understat").all()
    # The date spelling has to survive, or the next update cannot match rows.
    assert result["Date"].tolist() == ["12/08/2023", "13/08/2023"]


def test_backfill_adds_exactly_three_columns(league_dir, monkeypatch):
    """No duplicate or helper columns leak into the file."""
    path = write_season(league_dir, "E0.csv", base_rows())

    before = pd.read_csv(path).columns.tolist()

    monkeypatch.setattr(
        backfill_xg, "load_understat_data", lambda **kwargs: matching_understat()
    )

    backfill_xg.backfill_league_xg("PremierLeague")

    after = pd.read_csv(path).columns.tolist()

    assert after == [*before, *XG_COLUMNS, SOURCE_COLUMN]
    assert len(after) == len(set(after))


def test_backfill_never_overwrites_an_existing_value(league_dir, monkeypatch):
    """A hand-corrected xG survives, and keeps its own source label."""
    rows = base_rows()
    rows[0]["home_xg"] = 9.99
    rows[0][SOURCE_COLUMN] = "manual"

    path = write_season(league_dir, "E0.csv", rows)

    monkeypatch.setattr(
        backfill_xg, "load_understat_data", lambda **kwargs: matching_understat()
    )

    backfill_xg.backfill_league_xg("PremierLeague")

    result = pd.read_csv(path)

    assert result.loc[0, "home_xg"] == 9.99
    # A partly hand-entered row keeps the source it already claimed; crediting
    # Understat with the hand-entered value would be a false record.
    assert result.loc[0, SOURCE_COLUMN] == "manual"
    # The other side of that same match was still a gap, so it was filled.
    assert result.loc[0, "away_xg"] == 0.90


def test_backfill_is_idempotent(league_dir, monkeypatch):
    """A second run changes nothing and fills nothing."""
    path = write_season(league_dir, "E0.csv", base_rows())

    monkeypatch.setattr(
        backfill_xg, "load_understat_data", lambda **kwargs: matching_understat()
    )

    backfill_xg.backfill_league_xg("PremierLeague")
    first = path.read_bytes()

    summary = backfill_xg.backfill_league_xg("PremierLeague")

    assert path.read_bytes() == first
    assert summary["xg_filled"] == 0
    assert summary["xg_already_present"] == 2


def test_backfill_leaves_unmatched_rows_empty(league_dir, monkeypatch):
    """A fixture Understat never published keeps a null, not a zero."""
    rows = base_rows()
    rows.append({
        "Date": "20/08/2023",
        "HomeTeam": "Everton",
        "AwayTeam": "Fulham",
        "HS": 9,
        "AS": 11,
        "HST": 4,
        "AST": 5,
    })

    path = write_season(league_dir, "E0.csv", rows)

    monkeypatch.setattr(
        backfill_xg, "load_understat_data", lambda **kwargs: matching_understat()
    )

    summary = backfill_xg.backfill_league_xg("PremierLeague")

    result = pd.read_csv(path)

    assert pd.isna(result.loc[2, "home_xg"])
    assert pd.isna(result.loc[2, SOURCE_COLUMN])
    assert summary["rows_unmatched"] == 1
    assert len(result) == 3


def test_backfill_dry_run_writes_nothing(league_dir, monkeypatch):
    """A dry run reports the same numbers without touching the file."""
    path = write_season(league_dir, "E0.csv", base_rows())
    original = path.read_bytes()

    monkeypatch.setattr(
        backfill_xg, "load_understat_data", lambda **kwargs: matching_understat()
    )

    summary = backfill_xg.backfill_league_xg("PremierLeague", dry_run=True)

    assert path.read_bytes() == original
    assert summary["xg_filled"] == 2
    assert "home_xg" not in pd.read_csv(path).columns


def test_backfill_applies_the_team_name_map(league_dir, monkeypatch):
    """A club whose name Understat spells differently still matches.

    ``load_understat_data`` is what translates the names, so the stand-in here
    mimics it: it returns the football-data spelling, which is the contract the
    backfill matches on.
    """
    rows = base_rows()
    rows[1]["HomeTeam"] = "Wolves"

    path = write_season(league_dir, "E0.csv", rows)

    understat = understat_rows([
        {
            "date": "2023-08-13",
            "home_team": "Wolverhampton Wanderers",
            "away_team": "Everton",
            "home_xg": 1.75,
            "away_xg": 0.25,
        },
    ])
    understat["home_team"] = understat["home_team"].replace(TEAM_NAME_MAP)

    monkeypatch.setattr(
        backfill_xg, "load_understat_data", lambda **kwargs: understat
    )

    backfill_xg.backfill_league_xg("PremierLeague")

    assert pd.read_csv(path).loc[1, "home_xg"] == 1.75


def test_backfill_ignores_a_club_left_untranslated(league_dir, monkeypatch):
    """An unmapped name must not be matched on a guess.

    The Understat name is kept, the CSV spelling is not silently accepted, and
    the row is reported as unmatched so the gap is visible.
    """
    rows = base_rows()
    rows[1]["HomeTeam"] = "Wolves"

    path = write_season(league_dir, "E0.csv", rows)

    understat = understat_rows([
        {
            "date": "2023-08-13",
            "home_team": "Wolverhampton Wanderers",
            "away_team": "Everton",
            "home_xg": 1.75,
            "away_xg": 0.25,
        },
    ])

    monkeypatch.setattr(
        backfill_xg, "load_understat_data", lambda **kwargs: understat
    )

    summary = backfill_xg.backfill_league_xg("PremierLeague")

    assert pd.isna(pd.read_csv(path).loc[1, "home_xg"])
    assert summary["rows_unmatched"] == 2


def test_backfill_reads_a_file_it_wrote_itself(league_dir, monkeypatch):
    """Re-reading a backfilled file must not shift the dates."""
    rows = base_rows()
    path = write_season(league_dir, "E0.csv", rows)

    monkeypatch.setattr(
        backfill_xg, "load_understat_data", lambda **kwargs: matching_understat()
    )

    backfill_xg.backfill_league_xg("PremierLeague")
    before = pd.read_csv(path)["home_xg"].tolist()

    backfill_xg.backfill_league_xg("PremierLeague")

    assert pd.read_csv(path)["home_xg"].tolist() == before


def test_backfill_spans_every_season_of_the_league(league_dir, monkeypatch):
    """Each season file is handled, not just the first one."""
    write_season(
        league_dir, "E0.csv", [{**base_rows()[0], "Date": "12/08/2023"}]
    )
    second = write_season(
        league_dir, "E1.csv", [{**base_rows()[1], "Date": "11/08/2024"}]
    )

    understat = understat_rows([
        {
            "date": "2023-08-12",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "home_xg": 2.10,
            "away_xg": 0.90,
        },
        {
            "date": "2024-08-11",
            "home_team": "Liverpool",
            "away_team": "Everton",
            "home_xg": 2.80,
            "away_xg": 0.70,
        },
    ])

    requested = {}

    def record_request(**kwargs):
        requested.update(kwargs)
        return understat

    monkeypatch.setattr(
        backfill_xg, "load_understat_data", record_request
    )

    summary = backfill_xg.backfill_league_xg("PremierLeague")

    assert summary["files"] == 2
    assert summary["xg_filled"] == 2
    assert requested["start_year"] == 2023
    assert requested["end_year"] == 2024
    assert pd.read_csv(second).loc[0, "home_xg"] == 2.80


def test_backfill_writes_nothing_when_understat_is_unavailable(
    league_dir,
    monkeypatch,
):
    """A provider outage must not leave half a league backfilled."""

    def unavailable(**kwargs):
        raise UnderstatDataError("Understat returned nothing for PremierLeague")

    write_season(league_dir, "E0.csv", base_rows())
    second = write_season(league_dir, "E1.csv", base_rows())

    originals = {
        path: path.read_bytes() for path in (second,)
    }
    first = league_dir / "E0.csv"
    originals[first] = first.read_bytes()

    monkeypatch.setattr(
        backfill_xg, "load_understat_data", unavailable
    )

    with pytest.raises(UnderstatDataError):
        backfill_xg.backfill_league_xg("PremierLeague")

    for path, content in originals.items():
        assert path.read_bytes() == content


def test_backfill_reports_a_missing_directory(league_dir):
    """An empty league directory gets an actionable message."""
    with pytest.raises(FileNotFoundError, match="update_data.py"):
        backfill_xg.backfill_league_xg("PremierLeague")


def test_backfill_handles_an_empty_source_column(league_dir, monkeypatch):
    """An all-empty xg_source column comes back as float64, not text.

    pandas types a column of nothing but NaN as float64, and writing a string
    into a float column raises. Reading the file back is what produces that
    dtype, so the test reads the file it wrote rather than building the frame
    in memory.
    """
    path = write_season(league_dir, "E0.csv", base_rows())

    # Produce the float64 source column the way a real pre-backfill file has it.
    seed = pd.read_csv(path)
    seed[SOURCE_COLUMN] = np.nan
    seed.to_csv(path, index=False)

    monkeypatch.setattr(
        backfill_xg, "load_understat_data", lambda **kwargs: matching_understat()
    )

    backfill_xg.backfill_league_xg("PremierLeague")

    result = pd.read_csv(path)

    assert (result[SOURCE_COLUMN] == "understat").all()
    assert result["home_xg"].tolist() == [2.10, 2.80]
