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


def test_load_data_orders_same_day_matches_by_team(tmp_path, monkeypatch):
    """Test that matches sharing a date are ordered by the teams involved.

    A league plays several matches on the same day, and an unstable sort would
    leave their order up to the filesystem, which is not the same on every
    machine. The rolling features and the sequential Elo ratings are both built
    from the row order, so an unstable sort makes the model depend on the
    checkout.
    """
    league_path = tmp_path / "PremierLeague"
    league_path.mkdir()

    (league_path / "2023.csv").write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
        "01/08/2023,Team C,Team D,1,1,D\n"
        "01/08/2023,Team A,Team B,2,1,H\n"
        "01/08/2023,Team B,Team A,0,2,A\n"
    )

    monkeypatch.chdir(tmp_path)

    result = data_loader.load_data("PremierLeague")

    assert list(result["HomeTeam"]) == ["Team A", "Team B", "Team C"]


def test_load_data_ignores_the_order_files_are_read_in(tmp_path, monkeypatch):
    """Test that two directories holding the same matches load identically."""
    def write(directory, first, second):
        (directory / "PremierLeague").mkdir(parents=True)
        (directory / "PremierLeague" / "a.csv").write_text(
            "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n" + first
        )
        (directory / "PremierLeague" / "b.csv").write_text(
            "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n" + second
        )

    forward = tmp_path / "forward"
    reverse = tmp_path / "reverse"

    match_one = "05/08/2023,Team E,Team F,1,0,H\n"
    match_two = "05/08/2023,Team A,Team B,3,1,H\n"

    write(forward, match_one, match_two)
    write(reverse, match_two, match_one)

    monkeypatch.chdir(forward)
    first = data_loader.load_data("PremierLeague")

    monkeypatch.chdir(reverse)
    second = data_loader.load_data("PremierLeague")

    pd.testing.assert_frame_equal(first, second)


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


def test_write_csv_atomic_writes_the_frame(tmp_path):
    """Test that the frame is written with a header and no index."""
    path = tmp_path / "2025-2026.csv"
    frame = pd.DataFrame({"Date": ["01/09/2025"], "HomeTeam": ["Arsenal"]})

    data_loader.write_csv_atomic(path, frame)

    assert path.read_text(encoding="utf-8") == "Date,HomeTeam\n01/09/2025,Arsenal\n"


def test_write_csv_atomic_keeps_a_byte_order_mark(tmp_path):
    """Test that a file written with a BOM is rewritten with one.

    The season CSVs from football-data.co.uk carry a UTF-8 BOM, and dropping it
    would make Excel misread the file as Latin-1.
    """
    path = tmp_path / "2025-2026.csv"
    frame = pd.DataFrame({"HomeTeam": ["Atl\u00e9tico Madrid"]})

    data_loader.write_csv_atomic(path, frame, encoding="utf-8-sig")
    data_loader.write_csv_atomic(
        path,
        pd.DataFrame({"HomeTeam": ["Real Madrid"]}),
    )

    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert "Real Madrid" in path.read_text(encoding="utf-8-sig")


def test_write_csv_atomic_keeps_windows_line_endings(tmp_path):
    """Test that a CRLF file is not silently rewritten with LF endings."""
    path = tmp_path / "2025-2026.csv"
    frame = pd.DataFrame({"HomeTeam": ["Arsenal", "Chelsea"]})

    data_loader.write_csv_atomic(path, frame, line_terminator="\r\n")
    data_loader.write_csv_atomic(
        path,
        pd.DataFrame({"HomeTeam": ["Arsenal", "Chelsea", "Everton"]}),
    )

    raw = path.read_bytes()

    assert b"\r\n" in raw
    assert b"\n" not in raw.replace(b"\r\n", b"")


def test_write_csv_atomic_defaults_to_unix_endings_for_a_new_file(tmp_path):
    """Test that a season being created is not given Windows line endings."""
    path = tmp_path / "new.csv"

    data_loader.write_csv_atomic(path, pd.DataFrame({"HomeTeam": ["Arsenal"]}))

    assert b"\r\n" not in path.read_bytes()
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_write_csv_atomic_creates_missing_directories(tmp_path):
    """Test that the destination directory is created if it is absent."""
    path = tmp_path / "new" / "league" / "2025-2026.csv"

    data_loader.write_csv_atomic(path, pd.DataFrame({"HomeTeam": ["Arsenal"]}))

    assert path.exists()


def test_write_csv_atomic_leaves_no_temporary_file(tmp_path):
    """Test that the temporary file is moved rather than left behind."""
    path = tmp_path / "2025-2026.csv"

    data_loader.write_csv_atomic(path, pd.DataFrame({"HomeTeam": ["Arsenal"]}))

    assert list(tmp_path.glob("*.tmp")) == []


def test_write_csv_atomic_keeps_the_original_file_if_the_write_fails(
    tmp_path, monkeypatch
):
    """Test that a failure cannot truncate the existing season data."""
    path = tmp_path / "2025-2026.csv"
    original = "Date,HomeTeam\r\n01/09/2025,Arsenal\r\n"
    path.write_bytes(original.encode("utf-8-sig"))

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(data_loader.os, "replace", explode)

    with pytest.raises(OSError, match="disk full"):
        data_loader.write_csv_atomic(
            path, pd.DataFrame({"Date": ["02/09/2025"]})
        )

    assert path.read_bytes() == original.encode("utf-8-sig")


def test_get_season_range_spans_every_season_present():
    """
    Test that the season range covers the first and last season in the data.

    Deriving the range from the data is what keeps the Understat request in
    step with the CSVs, instead of a hardcoded range that falls behind.
    """
    dataframe = pd.DataFrame({
        "Date": pd.to_datetime([
            "2020-09-01",
            "2021-03-01",
            "2023-09-01",
        ]),
    })

    assert data_loader.get_season_range(dataframe) == (2020, 2023)


def test_get_season_range_uses_season_boundaries():
    """
    Test that a season is treated as running from July to June.

    A January fixture belongs to the season that began the previous July, so a
    naive min and max of the years would request a season that does not exist.
    """
    dataframe = pd.DataFrame({
        "Date": pd.to_datetime([
            "2026-01-01",
            "2026-08-01",
        ]),
    })

    assert data_loader.get_season_range(dataframe) == (2025, 2026)


def test_get_season_range_handles_single_season():
    """Test that a single season in the data yields a one year range."""
    dataframe = pd.DataFrame({
        "Date": pd.to_datetime(["2024-09-01", "2025-01-01"]),
    })

    assert data_loader.get_season_range(dataframe) == (2024, 2024)
