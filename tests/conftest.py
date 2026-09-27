"""
Shared test configuration.

The fixtures live here rather than in a test module so that both the
prediction tests and the command line tests can ask for them by name without
importing anything. A fixture imported into a test file is also invisible to
the linter, which sees the imported name as unused and offers to delete it.
"""

import pytest
from factories import build_rows, trained_predictor


@pytest.fixture
def league_dir(tmp_path, monkeypatch):
    """
    Builds a one-season PremierLeague directory in a temporary location.

    The predictor resolves its paths relative to the working directory, so the
    tests chdir rather than patch the config, which keeps the real path logic
    under test.

    Parameters:
        tmp_path (Path): A temporary directory provided by pytest.
        monkeypatch (pytest.MonkeyPatch): The test's patcher.

    Returns:
        Path: The league directory to write season files into.
    """
    directory = tmp_path / "football_data" / "PremierLeague"
    directory.mkdir(parents=True)

    monkeypatch.chdir(tmp_path)

    return directory


@pytest.fixture
def trained(league_dir):
    """
    A predictor over a played but incomplete season on disk.

    Enough matches are on disk for the ratings to mean something, and few
    enough pairings are played that unplayed fixtures still exist to ask for.

    Parameters:
        league_dir (Path): The league directory from league_dir.

    Returns:
        Predictor: A ready predictor, for tests that do not change the data.
    """
    return trained_predictor(league_dir, build_rows())
