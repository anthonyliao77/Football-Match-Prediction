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
    get_match_shots,
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


def shot_event(result, side="h"):
    """One shot from the per-match endpoint."""
    return {
        "id": "1",
        "result": result,
        "h_a": side,
        "xG": "0.05",
    }


def match_payload(home, away):
    """A per-match response in the shape the endpoint returns."""
    return {"shots": {"h": home, "a": away}, "rosters": {}, "tmpl": {}}


def test_get_match_shots_counts_total_shots(monkeypatch):
    install_session(monkeypatch, [FakeResponse(match_payload(
        [shot_event("Goal"), shot_event("BlockedShot"), shot_event("MissedShots")],
        [shot_event("SavedShot", "a")],
    ))])

    counts = get_match_shots("42")

    assert counts["home_shots"] == 3
    assert counts["away_shots"] == 1
    assert counts["home_on_target"] == 1
    assert counts["away_on_target"] == 1


def test_a_shot_off_the_woodwork_counts_as_on_target(monkeypatch):
    """The endpoint spells it ShotOnPost, and Post counts for nothing.

    Getting the token wrong undercounts by one on every match with a shot off
    the woodwork, which is a small silent error rather than a loud one."""
    install_session(monkeypatch, [FakeResponse(match_payload(
        [shot_event("Goal"), shot_event("ShotOnPost")],
        [],
    ))])

    counts = get_match_shots("42")

    assert counts["home_shots"] == 2
    assert counts["home_on_target"] == 2


def test_a_blocked_shot_is_not_on_target(monkeypatch):
    install_session(monkeypatch, [FakeResponse(match_payload(
        [shot_event("BlockedShot"), shot_event("MissedShots")],
        [],
    ))])

    counts = get_match_shots("42")

    assert counts["home_on_target"] == 0


def test_get_match_shots_asks_for_the_per_match_endpoint(monkeypatch):
    session = install_session(monkeypatch, [FakeResponse(
        match_payload([], [])
    )])

    get_match_shots("42")

    assert session.requested == ["https://understat.com/getMatchData/42"]


def test_a_response_with_no_shots_key_is_refused(monkeypatch):
    """A missing key is an endpoint that changed shape, not a match with no
    shots. Counting it as zero would write a made-up 0-0 into a real match."""
    install_session(monkeypatch, [FakeResponse({"rosters": {}})])

    with pytest.raises(UnderstatUnavailable, match="no shots key"):
        get_match_shots("42")


def test_a_goalless_match_really_can_have_no_shots(monkeypatch):
    """An empty list is a real answer and must not be refused.

    The distinction from the test above is the whole point: an empty list is a
    match nobody had a shot in, and it is the one input that has to survive."""
    install_session(monkeypatch, [FakeResponse(
        match_payload([], [])
    )])

    counts = get_match_shots("42")

    assert counts == {
        "home_shots": 0,
        "away_shots": 0,
        "home_on_target": 0,
        "away_on_target": 0,
    }


def test_the_two_league_readers_ask_the_endpoint_once_each(monkeypatch):
    """Both readers used to fetch the same URL for the same answer.

    get_league_season wanted the played matches and get_league_fixtures wanted
    the whole season, and those differ by which rows you keep rather than by
    which request you make, so every refresh was paying for the same payload
    twice per league season."""
    from src.understat_client import get_league_fixtures

    season = [played_match("2026-08-21"), future_match("2026-08-30")]

    session = install_session(monkeypatch, [FakeResponse({"dates": season})])

    get_league_season("EPL", "2026/2027")

    assert session.requested == ["https://understat.com/getLeagueData/EPL/2026"]

    session = install_session(monkeypatch, [FakeResponse({"dates": season})])

    get_league_fixtures("EPL", "2026/2027")

    assert session.requested == ["https://understat.com/getLeagueData/EPL/2026"]


def test_the_fixture_list_carries_the_id_the_shot_reader_needs(monkeypatch):
    """Understat's id is the only handle its per-match endpoint takes."""
    from src.understat_client import get_league_fixtures

    install_session(monkeypatch, [FakeResponse(
        {"dates": [played_match("2026-08-21"), future_match("2026-08-30")]}
    )])

    fixtures = get_league_fixtures("EPL", "2026/2027")

    assert fixtures["understat_id"].tolist() == ["1", "2"]


def test_a_fixture_with_no_id_reads_as_missing(monkeypatch):
    """A missing id is a gap in the response, not the empty string.

    The string would be truthy, and an empty id sent to the per-match endpoint
    would come back as whatever that path happens to serve."""
    from src.understat_client import get_league_fixtures

    match = played_match("2026-08-21")
    del match["id"]

    install_session(monkeypatch, [FakeResponse({"dates": [match]})])

    fixtures = get_league_fixtures("EPL", "2026/2027")

    assert pd.isna(fixtures["understat_id"].iloc[0])
