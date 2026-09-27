"""
Fetches expected-goals data directly from understat.com.

Understat publishes no API. The figures on the site are served by an
undocumented JSON endpoint that the page's own JavaScript calls, and this module
reads that endpoint directly.

Reading it directly rather than going through a library is deliberate. The
``soccerdata`` package discovers which seasons exist by asking Understat for a
league/season index and then inferring a season from the month of each row,
using ``season_id = year if month >= 7 else year - 1``. An in-progress season
only becomes visible once Understat publishes a row dated July or later, so a
season that is already being played stays invisible until then. That is how the
2026/2027 Premier League and Serie A data came to be missing while the La Liga
data was present: only La Liga had a qualifying row in the index at the time.
Asking for the season by name removes that dependency.
"""

import time

import pandas as pd
import requests

BASE_URL = "https://understat.com"

# Seconds to wait on a retryable response before trying again.
RETRY_BACKOFF_SECONDS = 1.0

# How many times a retryable request is attempted in total.
MAX_ATTEMPTS = 3

# Seconds to wait between two league seasons.
#
# Understat has no published rate limit, but it is a small free site with no
# API, and a full backfill asks for 21 of these. Half a second keeps the traffic
# negligible.
REQUEST_DELAY_SECONDS = 0.5

# The endpoint ignores requests that do not look like they came from the site's
# own page, and answers 404 rather than refusing them, so a missing header shows
# up as an empty season instead of an error.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

REQUEST_TIMEOUT = 30


class UnderstatUnavailable(RuntimeError):
    """Raised when a league season cannot be read from Understat."""


# One session is reused so the whole backfill shares a connection and a cookie
# jar instead of repeating the TCP and TLS handshake for each season.
_session = requests.Session()

# Understat only answers the JSON endpoint once the session has visited the
# site, so the cookie is fetched once and then left alone.
_cookies_ready = False

# When the last request went out, so consecutive ones can be spaced out.
_last_request_at = None


def _throttle() -> None:
    """
    Waits, if needed, so requests are not sent back to back.

    Applied to every request rather than to the season loop, so the spacing
    holds no matter how the client is called.
    """
    global _last_request_at

    if _last_request_at is not None:
        elapsed = time.monotonic() - _last_request_at

        if elapsed < REQUEST_DELAY_SECONDS:
            time.sleep(REQUEST_DELAY_SECONDS - elapsed)

    _last_request_at = time.monotonic()


def _ensure_cookies() -> None:
    """
    Visits the site once so the session holds a cookie the API will accept.

    Raises:
        UnderstatUnavailable: If the site cannot be reached at all.
    """
    global _cookies_ready

    if _cookies_ready:
        return

    try:
        _throttle()
        _session.get(BASE_URL, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as error:
        raise UnderstatUnavailable(
            f"Could not reach {BASE_URL}: {error}"
        ) from error

    _cookies_ready = True


def _get(url: str, referer: str) -> dict:
    """
    Fetches and parses one JSON payload, retrying transient failures.

    Parameters:
        url (str): The JSON endpoint to request.
        referer (str): The page the request is pretending to come from.

    Returns:
        dict: The decoded JSON body.

    Raises:
        UnderstatUnavailable: If the request keeps failing or the body is not
        the JSON the endpoint is documented to return.
    """
    _ensure_cookies()

    headers = {
        "User-Agent": BROWSER_USER_AGENT,
        "Referer": referer,
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "application/json, text/javascript, */*; q=0.01",
    }

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            _throttle()
            response = _session.get(
                url,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as error:
            if attempt == MAX_ATTEMPTS:
                raise UnderstatUnavailable(
                    f"Request to {url} failed: {error}"
                ) from error

            time.sleep(RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1))
            continue

        # 429 and 5xx are worth another go; a 404 is Understat saying it does not
        # have the season, and repeating that will not change the answer.
        retryable = (
            response.status_code == 429 or response.status_code >= 500
        )

        if retryable and attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1))
            continue

        if response.status_code != 200:
            raise UnderstatUnavailable(
                f"{url} returned HTTP {response.status_code}"
            )

        try:
            return response.json()
        except ValueError as error:
            raise UnderstatUnavailable(
                f"{url} did not return JSON, so the response format may have "
                f"changed: {error}"
            ) from error

    raise UnderstatUnavailable(f"{url} could not be read")


def _parse(payload: dict, league: str, season: str) -> pd.DataFrame:
    """
    Turns one league payload into a frame of played fixtures.

    Only finished matches are kept. A fixture that has not been played carries
    ``xG`` of ``{h: None, a: None}``, and a local fixture that has not been
    played either has no score, so pairing them would add a row that
    contributes nothing.

    Parameters:
        payload (dict): The decoded JSON body.
        league (str): The league name, for error messages.
        season (str): The season label, for error messages.

    Returns:
        pd.DataFrame: Columns date, home_team, away_team, home_xg and away_xg.

    Raises:
        UnderstatUnavailable: If the payload holds no matches at all.
    """
    matches = payload.get("dates") or []

    played = [match for match in matches if match.get("isResult")]

    if not played:
        raise UnderstatUnavailable(
            f"Understat has no played matches for {league} {season}. Either "
            f"the season has not started or Understat has not published it."
        )

    frame = pd.DataFrame([
        {
            "date": pd.to_datetime(match["datetime"]).normalize(),
            "home_team": match["h"]["title"],
            "away_team": match["a"]["title"],
            "home_xg": match["xG"]["h"],
            "away_xg": match["xG"]["a"],
        }
        for match in played
    ])

    for column in ("home_xg", "away_xg"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def get_league_season(slug: str, season: str) -> pd.DataFrame:
    """
    Reads one league season of expected-goals data.

    Parameters:
        slug (str): Understat's URL name for the league, e.g. "EPL".
        season (str): The season label, e.g. "2026/2027".

    Returns:
        pd.DataFrame: Columns date, home_team, away_team, home_xg and away_xg
        for every match Understat has a result for.

    Raises:
        UnderstatUnavailable: If the season cannot be read.
    """
    # The endpoint takes the start year alone: 2026/2027 is asked for as 2026.
    start_year = season.split("/")[0]

    page = f"{BASE_URL}/league/{slug}/{start_year}"
    payload = _get(
        f"{BASE_URL}/getLeagueData/{slug}/{start_year}",
        referer=page,
    )

    return _parse(payload, slug, season)


def get_league_fixtures(slug: str, season: str) -> pd.DataFrame:
    """
    Reads the whole fixture list for one league season.

    Unlike get_league_season, which returns only finished matches, this keeps
    the fixtures that have not been played yet. Understat publishes the season
    as it goes: a match that is still to be arrived at is already listed with
    its date and sides, and carries no score or xG until it is played. That is
    the difference between a schedule and a set of results, and reading it is
    how the remaining fixtures of the current season are picked up.

    Parameters:
        slug (str): Understat's URL name for the league, e.g. "EPL".
        season (str): The season label, e.g. "2026/2027".

    Returns:
        pd.DataFrame: Columns date, home_team, away_team, home_goals,
        away_goals, home_xg, away_xg and played, for every fixture listed. The
        score and xG columns are empty where the match is not yet played.

    Raises:
        UnderstatUnavailable: If the season cannot be read.
    """
    start_year = season.split("/")[0]

    page = f"{BASE_URL}/league/{slug}/{start_year}"
    payload = _get(
        f"{BASE_URL}/getLeagueData/{slug}/{start_year}",
        referer=page,
    )

    matches = payload.get("dates") or []

    if not matches:
        raise UnderstatUnavailable(
            f"Understat has no fixtures listed for {slug} {season}. Either "
            f"the season has not started or Understat has not published it."
        )

    frame = pd.DataFrame([
        {
            "date": pd.to_datetime(match["datetime"]).normalize(),
            "home_team": match["h"]["title"],
            "away_team": match["a"]["title"],
            "home_goals": (match.get("goals") or {}).get("h"),
            "away_goals": (match.get("goals") or {}).get("a"),
            "home_xg": (match.get("xG") or {}).get("h"),
            "away_xg": (match.get("xG") or {}).get("a"),
            "played": bool(match.get("isResult")),
        }
        for match in matches
    ])

    for column in ("home_goals", "away_goals", "home_xg", "away_xg"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame.sort_values("date").reset_index(drop=True)
