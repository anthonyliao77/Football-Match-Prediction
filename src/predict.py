"""
Predicts the outcome of a fixture that has not been played yet.

The training pipeline is chronological. Every feature it builds describes the
matches that came *before* a fixture, and the Elo rating it uses is the one each
team held going into that match. So predicting an unplayed fixture needs no new
feature code: the fixture is appended to the data as a synthetic row with no
result, the existing feature pass is run over the frame, and that row's features
are read back.

Going through the same pass is the point. A separately written path that
recomputed the rolling window for one fixture would be free to drift from the
one the model was trained and scored against, and the drift would only show up
as slightly disappointing accuracy much later. Here a prediction cannot disagree
with the training pipeline, because there is nothing to disagree with.

The synthetic row's result, score and xG are left blank rather than invented.
None of them are read for that row, because a feature describes what happened
before a match. Leaving them empty is load-bearing rather than tidy: a fixture
asked for on a date that other fixtures share lands in the middle of the frame,
so an invented 0-0 would be read as a real goalless draw by every row after it,
poisoning their rolling windows and Elo. Empty says "not played yet", and the
feature pass already knows what to do with that.
"""

import difflib
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier

from config import FEATURE_COLUMNS, LEAGUES
from src.data_loader import (
    load_data,
    played_matches,
    prediction_season,
    previous_season,
)
from src.elo import (
    HOME_ADVANTAGE,
    calculate_expected_score,
    create_elo_features,
    get_season,
)
from src.features import create_features

# The columns create_features reads off each match when it extends a team's
# rolling window. They are blanked on the synthetic row, and nothing else has to
# be, because every other column is either an identifier or a betting price that
# no feature reads.
STATISTIC_COLUMNS = (
    "FTR",
    "FTHG",
    "FTAG",
    "HS",
    "AS",
    "HST",
    "AST",
    "home_xg",
    "away_xg",
)

# Below this many previous matches a team's rolling window and Elo rating are
# too thin to carry much weight. A club promoted into the current season has
# three or four, so its rating is barely off the 1500 starting value and would
# otherwise be presented as if it were established.
THIN_HISTORY_MATCHES = 10

# How far ahead of the last played match a fixture is assumed to sit when no
# date is given. A week is the usual gap between matchdays.
DEFAULT_DATE_OFFSET_DAYS = 7

# The order outcomes are reported in, which is not the order a classifier
# returns them in.
OUTCOME_ORDER = ("H", "D", "A")

OUTCOME_LABELS = {"H": "Home", "D": "Draw", "A": "Away"}

MODEL_LABELS = {"rf": "Random Forest", "xgb": "XGBoost"}

# Spellings accepted for a league beyond the keys of LEAGUES, so that the name
# a person actually types does not have to be the directory name.
LEAGUE_ALIASES = {
    "epl": "PremierLeague",
    "premier league": "PremierLeague",
    "premierleague": "PremierLeague",
    "laliga": "LaLiga",
    "la liga": "LaLiga",
    "la_liga": "LaLiga",
    "serie a": "SerieA",
    "seriea": "SerieA",
}

# The feature each form figure is read from, and how it is labelled. The form
# summary is the feature vector under shorter names rather than a second
# calculation, so the two can never disagree.
FORM_FEATURES = (
    ("PT5", "pts"),
    ("GS5", "scored"),
    ("GC5", "conceded"),
    ("XG5", "xG"),
)


class PredictionError(RuntimeError):
    """Raised when a fixture cannot be predicted."""


class UnknownLeagueError(PredictionError):
    """Raised when the league is not one this project covers."""


class UnknownTeamError(PredictionError):
    """Raised when a team does not appear in the league's data."""


class AlreadyPlayedError(PredictionError):
    """Raised when the data already reaches the fixture being predicted."""


@dataclass
class Prediction:
    """
    The outcome of one unplayed fixture, and the state it was predicted from.

    Attributes:
        league (str): The league directory name.
        home (str): The home team, as spelled in the CSVs.
        away (str): The away team, as spelled in the CSVs.
        date (pd.Timestamp): When the fixture is assumed to be played.
        data_through (pd.Timestamp): The date of the most recent match in the
            data, which is the last thing the prediction can know about.
        matches (int): Rows of match data behind the prediction.
        teams (int): Distinct clubs in that data.
        probabilities (dict): Outcome probabilities keyed by model label, then
            by outcome letter.
        outcomes (dict): The most likely outcome per model label.
        elo (dict): Pre-match Elo ratings keyed by "home" and "away".
        elo_expected (dict): What the ratings alone expect, keyed the same way.
        form (dict): Last-five figures per side, keyed by "home" and "away".
        features (pd.Series): The full feature vector, for --explain.
        notes (list): Anything a reader should know before trusting it.
    """

    league: str
    home: str
    away: str
    date: pd.Timestamp
    data_through: pd.Timestamp
    matches: int
    teams: int
    probabilities: dict = field(default_factory=dict)
    outcomes: dict = field(default_factory=dict)
    elo: dict = field(default_factory=dict)
    elo_expected: dict = field(default_factory=dict)
    form: dict = field(default_factory=dict)
    features: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    notes: list = field(default_factory=list)


def resolve_league(name: str) -> str:
    """
    Finds the league a name refers to.

    Parameters:
        name (str): A LEAGUES key, or a common spelling of one.

    Returns:
        str: The matching LEAGUES key.

    Raises:
        UnknownLeagueError: If the name matches no league.
    """
    cleaned = name.strip()

    if cleaned in LEAGUES:
        return cleaned

    resolved = LEAGUE_ALIASES.get(cleaned.lower())

    if resolved is None:
        raise UnknownLeagueError(
            f"Unknown league '{name}'. Choose one of: "
            f"{', '.join(LEAGUES)}."
        )

    return resolved


class Predictor:
    """
    A trained model for one league, able to predict fixtures not yet played.

    The data and both models are built once on construction, so a session that
    predicts several fixtures fits each model once for the whole run instead of
    once per fixture. The feature pass is not shared that way, and cannot be:
    it reads the data plus the fixture being asked about, so that frame differs
    every time.

    Attributes:
        league (str): The league directory name.
        dataframe (pd.DataFrame): Every match in the league, in date order.
        teams (list): The clubs that appear in that data.
        data_through (pd.Timestamp): The date of the most recent match.
        appearances (Counter): How many matches each club appears in.
    """

    def __init__(self, league: str):
        """
        Loads the league, builds its features and fits both models.

        The models are fitted only on the seasons before the validation season.
        The season being predicted is in neither the fit nor the validation
        split, so the scores src/training.py reports describe this model rather
        than a differently scoped one.

        The rolling features and Elo ratings used to place a fixture are built
        from every match played so far, including the season being predicted.
        A result that has already happened is real information about how a team
        is playing, and withholding it would only make the prediction worse.

        Parameters:
            league (str): The league to prepare, as a LEAGUES key or alias.

        Raises:
            UnknownLeagueError: If the league is not one this project covers.
        """
        self.league = resolve_league(league)

        self.dataframe = played_matches(
            load_data(LEAGUES[self.league]["football_data"])
        )

        self.teams = sorted(
            set(self.dataframe["HomeTeam"]) | set(self.dataframe["AwayTeam"])
        )

        self.data_through = self.dataframe["Date"].max()

        self.season_under_prediction = prediction_season(self.dataframe)

        self.validation_season = previous_season(self.season_under_prediction)

        self.appearances = Counter()

        for _, match in self.dataframe.iterrows():
            self.appearances[match["HomeTeam"]] += 1
            self.appearances[match["AwayTeam"]] += 1

        prepared = create_elo_features(create_features(self.dataframe))

        # The fit stops at the training boundary. A season at or after the
        # validation season is left out, so a fixture in the season being
        # predicted can never be one of the model's own training rows.
        boundary = int(self.validation_season.split("/")[0])

        training = prepared[
            prepared["Date"].map(get_season).str[:4].astype(int) < boundary
        ]

        self.training_matches = len(training)

        self._random_forest = self._fit_random_forest(training)
        self._xgb, self._encoder = self._fit_xgb(training)

    def _fit_random_forest(self, prepared: pd.DataFrame):
        """
        Fits the Random Forest on the training seasons.

        The hyperparameters are the ones src/training.py reports scores for, and
        the training seasons are the ones it fits on, so a prediction comes from
        the model that was evaluated rather than a differently configured one.

        Parameters:
            prepared (pd.DataFrame): The training rows of the feature frame.

        Returns:
            RandomForestClassifier: The fitted model.
        """
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=67,
            n_jobs=-1,
        )

        model.fit(prepared[FEATURE_COLUMNS], prepared["FTR"])

        return model

    def _fit_xgb(self, prepared: pd.DataFrame):
        """
        Fits XGBoost on the training seasons, with its labels encoded.

        Parameters:
            prepared (pd.DataFrame): The training rows of the feature frame.

        Returns:
            tuple: The fitted model and the encoder that maps its integer
            labels back to outcome letters.
        """
        encoder = LabelEncoder()

        encoded = encoder.fit_transform(prepared["FTR"])

        model = XGBClassifier(
            n_estimators=200,
            random_state=67,
            learning_rate=0.01,
            n_jobs=-1,
        )

        model.fit(prepared[FEATURE_COLUMNS], encoded)

        return model, encoder

    def resolve_team(self, name: str) -> str:
        """
        Finds the CSV spelling of a team from what a person typed.

        Parameters:
            name (str): A team name, in any case.

        Returns:
            str: The spelling used in the season CSVs.

        Raises:
            UnknownTeamError: If no team matches, with the closest names.
        """
        cleaned = name.strip()

        for team in self.teams:
            if team.lower() == cleaned.lower():
                return team

        suggestions = difflib.get_close_matches(cleaned, self.teams, n=3)

        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""

        raise UnknownTeamError(
            f"'{cleaned}' has not played in the {self.league} data."
            f"{hint}"
        )

    def default_date(self) -> pd.Timestamp:
        """
        The date used when none is given.

        Returns:
            pd.Timestamp: A week after the last match in the data.
        """
        return self.data_through + pd.Timedelta(days=DEFAULT_DATE_OFFSET_DAYS)

    def _parse_date(self, value) -> pd.Timestamp:
        """
        Reads a fixture date from a string or a timestamp.

        Parameters:
            value (str | pd.Timestamp): The date the fixture is played.

        Returns:
            pd.Timestamp: The normalised date.

        Raises:
            PredictionError: If the value cannot be read as a date.
        """
        if isinstance(value, pd.Timestamp):
            return value.normalize()

        try:
            return pd.to_datetime(value).normalize()
        except (ValueError, TypeError) as error:
            raise PredictionError(
                f"Could not read '{value}' as a date. Use YYYY-MM-DD."
            ) from error

    def _check_fixture(self, home: str, away: str, date: pd.Timestamp) -> None:
        """
        Rejects fixtures the model cannot honestly be asked about.

        Parameters:
            home (str): The home team, as spelled in the CSVs.
            away (str): The away team, as spelled in the CSVs.
            date (pd.Timestamp): The date the fixture is to be played.

        Raises:
            PredictionError: If both sides are the same team.
            AlreadyPlayedError: If the fixture is already in the data, or the
                data already reaches past the date asked for.
        """
        if home == away:
            raise PredictionError(
                f"{home} cannot play itself. Check the home and away teams."
            )

        # Only a meeting at or after the requested date rules the fixture out.
        # Two clubs having met before is the normal state of affairs, and the
        # next meeting between them is usually exactly the thing being asked
        # about, so a past meeting is not by itself a reason to refuse.
        played = self.dataframe[
            (self.dataframe["HomeTeam"] == home)
            & (self.dataframe["AwayTeam"] == away)
            & (self.dataframe["Date"] >= date)
        ]

        if played.empty:
            return

        latest = played.iloc[-1]

        if latest["Date"] == date:
            raise AlreadyPlayedError(
                f"{home} v {away} has already been played on "
                f"{latest['Date'].date()}: "
                f"{latest['FTHG']}-{latest['FTAG']} (FTR {latest['FTR']}). "
                f"This tool predicts fixtures that are still ahead of the "
                f"data."
            )

        raise AlreadyPlayedError(
            f"{home} v {away} has already been played, on "
            f"{latest['Date'].date()}, after the date asked for "
            f"({date.date()}). Check the date."
        )

    def _fixture_frame(self, home: str, away: str, date: pd.Timestamp):
        """
        Appends the unplayed fixture to the data as a synthetic row.

        The row is cloned from the last real match so it carries every column
        the frame expects, including the hundred or so betting odds that no
        feature reads. Only the result, score and xG columns are blanked.

        Parameters:
            home (str): The home team.
            away (str): The away team.
            date (pd.Timestamp): When the fixture is played.

        Returns:
            tuple: The extended frame in the order the feature pass will read
            it, and the position of the synthetic row within it.
        """
        row = self.dataframe.iloc[-1].copy()

        row["Date"] = date
        row["HomeTeam"] = home
        row["AwayTeam"] = away

        for column in STATISTIC_COLUMNS:
            row[column] = np.nan

        extended = pd.concat(
            [self.dataframe, pd.DataFrame([row])],
            ignore_index=True,
        )

        ordered = extended.sort_values(
            ["Date", "HomeTeam", "AwayTeam"]
        )

        # The synthetic row keeps its label through the sort, so its position
        # is looked up rather than assumed to be last. A fixture sharing a date
        # with real matches lands in the middle of that day, and taking the last
        # row there would return the wrong fixture's features.
        position = ordered.index.get_indexer([len(self.dataframe)])[0]

        return ordered, position

    def _notes_for(
        self,
        home: str,
        away: str,
        date: pd.Timestamp,
        features: pd.Series,
    ) -> list:
        """
        Collects the caveats that belong with this particular prediction.

        Parameters:
            home (str): The home team.
            away (str): The away team.
            date (pd.Timestamp): When the fixture is played.
            features (pd.Series): The feature vector built for the fixture.

        Returns:
            list: Human-readable notes, empty when nothing needs saying.
        """
        notes = []

        for side, team in (("home", home), ("away", away)):
            seen = self.appearances[team]

            if seen < THIN_HISTORY_MATCHES:
                notes.append(
                    f"{team} has {seen} matches in the data, so its form is "
                    f"built from a short window and its Elo rating is still "
                    f"close to the 1500 starting value."
                )

        if date < self.data_through:
            notes.append(
                f"The fixture date is before the last match in the data "
                f"({self.data_through.date()}), so the form and ratings are "
                f"those of that date and not today's."
            )
        elif get_season(date) != get_season(self.data_through):
            notes.append(
                f"{date.date()} falls in {get_season(date)}, a season the "
                f"data has not reached, so both ratings have been regressed "
                f"towards the mean as they are at the start of a season."
            )

        xg_features = [f"{side}XG5" for side in ("Home", "Away")]
        xg_features += [f"{side}XGA5" for side in ("Home", "Away")]

        if sum(abs(float(features[column])) for column in xg_features) == 0:
            notes.append(
                "Neither side has measured xG in its recent window, so the "
                "four xG features are all zero and the model is working "
                "without them."
            )

        return notes

    def _probabilities(self, model, features: pd.Series, encoder=None) -> dict:
        """
        Reads one model's outcome probabilities.

        The columns are matched to outcomes through the model's own classes_
        rather than by position. The classes come back in whatever order the
        label set sorted into, and a hardcoded column index is only correct
        while that order happens to be A, D, H.

        Parameters:
            model: A fitted classifier exposing predict_proba and classes_.
            features (pd.Series): The feature vector for one fixture.
            encoder (LabelEncoder | None): Present when the model's labels are
                encoded integers rather than letters.

        Returns:
            dict: Probabilities keyed by outcome letter.
        """
        # The feature vector is a single row taken out of a frame that mixes
        # ints and floats, so selecting it yields an object Series and every
        # column of the transposed frame would come back as object. XGBoost
        # rejects that outright. The features are numbers, so they are stated
        # as numbers.
        frame = features[FEATURE_COLUMNS].astype(float).to_frame().T

        probabilities = model.predict_proba(frame)[0]

        classes = model.classes_

        if encoder is not None:
            classes = encoder.inverse_transform(classes)

        return {
            str(label): float(value)
            for label, value in zip(classes, probabilities)
        }

    def predict(self, home: str, away: str, date=None) -> Prediction:
        """
        Predicts one unplayed fixture.

        Parameters:
            home (str): The home team, as spelled in the CSVs.
            away (str): The away team, as spelled in the CSVs.
            date (str | pd.Timestamp | None): When the fixture is played.
                Defaults to a week after the most recent match in the data.

        Returns:
            Prediction: The outcome probabilities, the state behind them, and
            any caveats.

        Raises:
            PredictionError: If the date cannot be read, or the fixture is not
                one the model can honestly be asked about.
            AlreadyPlayedError: If the data already reaches the requested
                date, which means the fixture is decided or has happened.
        """
        if date is None or (isinstance(date, str) and not date.strip()):
            fixture_date = self.default_date()
        else:
            fixture_date = self._parse_date(date)

        self._check_fixture(home, away, fixture_date)

        ordered, position = self._fixture_frame(home, away, fixture_date)

        prepared = create_elo_features(create_features(ordered))

        features = prepared.iloc[position][FEATURE_COLUMNS]

        random_forest = self._probabilities(self._random_forest, features)

        xgb = self._probabilities(self._xgb, features, self._encoder)

        home_elo = float(features["HomeEloBefore"])
        away_elo = float(features["AwayEloBefore"])

        return Prediction(
            league=self.league,
            home=home,
            away=away,
            date=fixture_date,
            data_through=self.data_through,
            matches=len(self.dataframe),
            teams=len(self.teams),
            probabilities={
                MODEL_LABELS["rf"]: random_forest,
                MODEL_LABELS["xgb"]: xgb,
            },
            outcomes={
                MODEL_LABELS["rf"]: max(
                    random_forest, key=random_forest.get
                ),
                MODEL_LABELS["xgb"]: max(xgb, key=xgb.get),
            },
            elo={"home": home_elo, "away": away_elo},
            elo_expected={
                "home": calculate_expected_score(
                    home_elo + HOME_ADVANTAGE, away_elo
                ),
                "away": calculate_expected_score(
                    away_elo, home_elo + HOME_ADVANTAGE
                ),
            },
            form={
                "home": _form_of(features, "Home"),
                "away": _form_of(features, "Away"),
            },
            features=features,
            notes=self._notes_for(home, away, fixture_date, features),
        )


def _form_of(features: pd.Series, side: str) -> dict:
    """
    Reads one side's last-five figures out of a feature vector.

    Parameters:
        features (pd.Series): The feature vector for one fixture.
        side (str): "Home" or "Away".

    Returns:
        dict: The figures, keyed by their short labels.
    """
    return {
        label: features[f"{side}{suffix}"]
        for suffix, label in FORM_FEATURES
    }


def format_prediction(
    prediction: Prediction,
    explain: bool = False,
    models: list | None = None,
) -> str:
    """
    Renders a prediction as the block a person reads.

    The Elo figures are printed next to the model output deliberately. The two
    come from different evidence, and when they disagree the disagreement is
    the most useful thing on screen.

    Parameters:
        prediction (Prediction): The prediction to render.
        explain (bool): If True, append the full feature vector.
        models (list | None): Model labels to show, or None for all of them.

    Returns:
        str: The rendered block.
    """
    # The multi-line strings are wrapped in parentheses so that the
    # concatenation is stated rather than implied. Inside a list, a missing
    # comma between two adjacent strings is a silent join, and these lines are
    # exactly the shape where that mistake is invisible.
    lines = [
        (
            f"{prediction.league} - {prediction.home} v "
            f"{prediction.away}, {prediction.date:%a %d %b %Y}"
        ),
        (
            f"Data through {prediction.data_through:%Y-%m-%d} "
            f"({prediction.matches} matches, {prediction.teams} teams)"
        ),
        "",
    ]

    shown = [
        label
        for label in prediction.probabilities
        if models is None or label in models
    ]

    width = max(len(label) for label in shown)

    for label in shown:
        probabilities = prediction.probabilities[label]
        shares = "  ".join(
            f"{OUTCOME_LABELS[outcome]} {probabilities[outcome] * 100:>3.0f}%"
            for outcome in OUTCOME_ORDER
        )

        outcome = prediction.outcomes[label]

        lines.append(
            f"  {label:<{width}}  {shares}   ->  "
            f"{OUTCOME_LABELS[outcome].upper()}"
        )

    home_elo = prediction.elo["home"]
    away_elo = prediction.elo["away"]

    lines += [
        "",
        (
            f"  Elo            {home_elo:>7.0f}  vs  {away_elo:>7.0f}"
            f"     (home advantage +{HOME_ADVANTAGE})"
        ),
        (
            f"  Elo expectancy {prediction.elo_expected['home']:>7.2f}"
            f"  vs  {prediction.elo_expected['away']:>7.2f}"
            f"     (ratings only, no draw model)"
        ),
    ]

    for side, team in (("home", prediction.home), ("away", prediction.away)):
        figures = "  ".join(
            f"{value:>6.2f}" if isinstance(value, float) else f"{value:>6}"
            for value in prediction.form[side].values()
        )

        lines.append(f"  Form, last 5   {team:<14}{figures}")

    lines.append("                 " + "  ".join(
        label.rjust(6) for _, label in FORM_FEATURES
    ))

    if explain:
        lines += ["", "  Feature vector:"]

        for column in FEATURE_COLUMNS:
            lines.append(f"    {column:<14} {prediction.features[column]:.4f}")

    if prediction.notes:
        lines += ["", "  Notes:"]

        lines += [f"    - {note}" for note in prediction.notes]

    return "\n".join(lines)
