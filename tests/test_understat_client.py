"""
Tests for the direct Understat client.
"""

import pandas as pd
import pytest

from src.understat_client import (
    REQUEST_DELAY_SECONDS,
    UnderstatUnavailable,
    _parse,
    get_league_season,
)


class FakeResponse:
    """
    Stands in for a requests.Response.
    """

    def __init__(self, payload=None, status_code=200, raises=None):
        self.payload = payload
        self.status_code = status_code
        self.raises = raises
        self.text = "" if payload is None else "body"

    def json(self):
        if self.raises is not None:
            raise self.raises

        return self.payload


class FakeSession:
    """
    Records the URLs requested and replays queued responses.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.requested = []
        self.cookies = {}

    def get(self, url, headers=None, timeout=None):
        self.requested.append(url)

        if not self.responses:
            raise AssertionError(f"unexpected request to {url}")

        return self.responses.pop(0)


def played_match(day, home="Manchester City", away="Arsenal", home_xg="1.85",
                 away_xg="0.56"):
    """A finished fixture in the shape the endpoint returns."""
    return {
        "id": "1",
        "datetime": f"{day} 21:00",
        "h": {"title": home},
        "a": {"title": away},
        "goals": {"h": "3", "a": "0"},
        "xG": {"h": home_xg, "a": away_xg},
        "isResult": True,
    }


def future_match(day):
    """A fixture that has not been played, which carries no xG."""
    return {
        "id": "2",
        "datetime": f"{day} 21:00",
        "h": {"title": "Liverpool"},
        "a": {"title": "Everton"},
        "goals": {"h": None, "a": None},
        "xG": {"h": None, "a": None},
        "isResult": False,
    }


@pytest.fixture(autouse=True)
def recorded_sleeps(monkeypatch):
    """
    Keeps the retry and throttle tests instant, and records what was asked for.

    The waiting is real, but the tests are checking that it happens and not how
    long anyone sat through it.
    """
    slept = []

    monkeypatch.setattr(
        "src.understat_client.time.sleep", lambda seconds: slept.append(seconds)
    )
    monkeypatch.setattr("src.understat_client._last_request_at", None)

    return slept


def install_session(monkeypatch, responses, ready=True):
    """
    Points the client at a fake session and marks the cookie as fetched.
    """
    session = FakeSession(responses)

    monkeypatch.setattr("src.understat_client._session", session)
    monkeypatch.setattr("src.understat_client._cookies_ready", ready)

    return session


def test_parse_keeps_played_matches_only():
    """A fixture with no result has no xG, so it is dropped."""
    frame = _parse(
        {"dates": [played_match("2026-08-21"), future_match("2026-10-10")]},
        "EPL",
        "2026/2027",
    )

    assert len(frame) == 1
    assert frame.loc[0, "home_team"] == "Manchester City"
    assert frame.loc[0, "away_team"] == "Arsenal"


def test_parse_converts_xg_to_numbers():
    """The endpoint sends xG as strings, and the frame needs floats."""
    frame = _parse(
        {"dates": [played_match("2026-08-21", home_xg="1.85424")]},
        "EPL",
        "2026/2027",
    )

    assert frame.loc[0, "home_xg"] == pytest.approx(1.85424)
    assert isinstance(frame.loc[0, "away_xg"], float)


def test_parse_normalises_the_date():
    """The kick-off time is dropped so the date can be matched on."""
    frame = _parse({"dates": [played_match("2026-08-21 21:00")]}, "EPL", "2026/2027")

    assert frame.loc[0, "date"] == pd.Timestamp("2026-08-21")


def test_parse_rejects_a_season_with_nothing_played():
    """An empty season is named, not returned as a silent empty frame."""
    with pytest.raises(UnderstatUnavailable, match="EPL 2026/2027"):
        _parse({"dates": [future_match("2026-10-10")]}, "EPL", "2026/2027")


def test_parse_rejects_a_payload_that_lost_its_shape():
    """A response with no matches array is an error, not a silent zero rows."""

    with pytest.raises(UnderstatUnavailable, match="EPL 2026/2027"):
        _parse({}, "EPL", "2026/2027")


def test_get_league_season_asks_for_the_start_year(monkeypatch):
    """2026/2027 is requested as 2026, which is what the endpoint takes."""
    session = install_session(
        monkeypatch,
        [FakeResponse({"dates": [played_match("2026-08-21")]})],
    )

    get_league_season("EPL", "2026/2027")

    assert session.requested == [
        "https://understat.com/getLeagueData/EPL/2026"
    ]


def test_get_league_season_visits_the_site_first(monkeypatch):
    """The JSON endpoint is only answered once the session holds a cookie."""
    session = install_session(
        monkeypatch,
        [
            FakeResponse({}),
            FakeResponse({"dates": [played_match("2026-08-21")]}),
        ],
        ready=False,
    )

    get_league_season("EPL", "2026/2027")

    assert session.requested[0] == "https://understat.com"
    assert "getLeagueData" in session.requested[1]


def test_the_first_request_is_not_delayed(monkeypatch, recorded_sleeps):
    """Nothing to be spaced out from on a cold start."""
    install_session(
        monkeypatch,
        [FakeResponse({"dates": [played_match("2026-08-21")]})],
    )

    get_league_season("EPL", "2026/2027")

    assert recorded_sleeps == []


def test_consecutive_requests_are_spaced_out(monkeypatch, recorded_sleeps):
    """A second season waits, so a full backfill is not a burst of traffic."""
    install_session(
        monkeypatch,
        [
            FakeResponse({"dates": [played_match("2025-08-16")]}),
            FakeResponse({"dates": [played_match("2026-08-21")]}),
        ],
    )

    get_league_season("EPL", "2025/2026")
    get_league_season("EPL", "2026/2027")

    assert len(recorded_sleeps) == 1
    assert recorded_sleeps[0] == pytest.approx(REQUEST_DELAY_SECONDS, abs=0.01)


def test_get_league_season_raises_on_a_404(monkeypatch):
    """A 404 means Understat does not have the season, and is not retried."""
    session = install_session(monkeypatch, [FakeResponse(status_code=404)])

    with pytest.raises(UnderstatUnavailable, match="HTTP 404"):
        get_league_season("EPL", "2026/2027")

    assert len(session.requested) == 1


def test_get_league_season_retries_a_server_error(monkeypatch):
    """A 500 is transient, so it gets another attempt before giving up."""
    install_session(
        monkeypatch,
        [
            FakeResponse(status_code=500),
            FakeResponse({"dates": [played_match("2026-08-21")]}),
        ],
    )

    frame = get_league_season("EPL", "2026/2027")

    assert len(frame) == 1


def test_get_league_season_gives_up_after_the_last_attempt(monkeypatch):
    """Repeated server errors end in a named failure, not an empty frame."""
    install_session(
        monkeypatch,
        [FakeResponse(status_code=503) for _ in range(3)],
    )

    with pytest.raises(UnderstatUnavailable, match="HTTP 503"):
        get_league_season("EPL", "2026/2027")


def test_get_league_season_rejects_a_non_json_body(monkeypatch):
    """If the site starts serving something else, say so plainly."""

    class BadJSON(ValueError):
        pass

    install_session(
        monkeypatch,
        [FakeResponse(raises=BadJSON("not json"))],
    )

    with pytest.raises(UnderstatUnavailable, match="did not return JSON"):
        get_league_season("EPL", "2026/2027")
