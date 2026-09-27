"""
Tests for predicting fixtures that have not been played yet.

The prediction path has one job that matters more than the probabilities it
returns: it must produce the same feature vector the training pipeline would
have produced for that fixture. A prediction built any other way could look
entirely reasonable and still be scored against a model it was never shaped
for, and the only symptom would be accuracy that is quietly worse than it
should be. The first two tests below are the ones that would catch that.
"""

import numpy as np
import pandas as pd
import pytest
from factories import (
    build_rows,
    trained_predictor,
    unplayed_pair,
    write_season,
)

from src.data_loader import load_data
from src.elo import A_FACTOR, create_elo_features
from src.features import create_features
from src.predict import (
    FEATURE_COLUMNS,
    STATISTIC_COLUMNS,
    AlreadyPlayedError,
    PredictionError,
    Predictor,
    UnknownLeagueError,
    UnknownTeamError,
    format_prediction,
    resolve_league,
)


def without_row(directory, rows, index):
    """
    Rewrites the season file with one row removed and the rest left in place.
    """
    kept = [row for position, row in enumerate(rows) if position != index]

    pd.DataFrame(kept).to_csv(
        directory / "2024-2025.csv", index=False
    )


def pick_row(rows, index):
    """
    Reads a fixture's teams and date out of a row set.
    """
    row = rows[index]

    date = pd.to_datetime(row["Date"], format="%d/%m/%Y")

    return row["HomeTeam"], row["AwayTeam"], date


def truth_features(home, away, date):
    """
    Reads the feature vector the training pipeline gave one fixture.

    The row is found by identity rather than by position. The pipeline sorts
    the frame by date and then by team, so a fixture's index in the sorted
    frame is not the index it had in the file it was written from.
    """
    prepared = create_elo_features(
        create_features(load_data("football_data/PremierLeague"))
    )

    match = prepared[
        (prepared["HomeTeam"] == home)
        & (prepared["AwayTeam"] == away)
        & (prepared["Date"] == date)
    ]

    assert len(match) == 1, "the fixture should appear exactly once"

    return match.iloc[0]


def test_prediction_reproduces_the_training_features(league_dir):
    """
    A fixture predicted as unplayed gets the features training would give it.

    One match is taken out of the data and asked for by name. Every feature
    must match what the pipeline produced for that row while it was still
    there, because that is the vector the model was fitted on.
    """
    rows = build_rows()
    write_season(league_dir, rows)

    home, away, date = pick_row(rows, 25)

    truth = truth_features(home, away, date)

    without_row(league_dir, rows, 25)

    prediction = Predictor("PremierLeague").predict(home, away, date)

    for column in FEATURE_COLUMNS:
        assert prediction.features[column] == pytest.approx(
            float(truth[column])
        ), column


def test_features_survive_a_fixture_that_is_not_the_last_row(league_dir):
    """
    A fixture with later matches still in the data is read correctly.

    Removing one match leaves the rest of the season after it, so the fixture
    lands in the middle of the frame. Taking the last row instead of the row's
    own position would hand back a different match's features, and every
    assertion above would pass for the wrong reason.
    """
    rows = build_rows()
    write_season(league_dir, rows)

    home, away, date = pick_row(rows, 25)

    truth = truth_features(home, away, date)

    without_row(league_dir, rows, 25)

    predictor = Predictor("PremierLeague")

    ordered, position = predictor._fixture_frame(home, away, date)

    assert position != len(ordered) - 1, "the fixture must not be the last row"
    assert ordered.iloc[position]["HomeTeam"] == home

    # The same fixture, read through the real entry point.
    prediction = predictor.predict(home, away, date)

    for column in FEATURE_COLUMNS:
        assert prediction.features[column] == pytest.approx(
            float(truth[column])
        ), column


def test_the_synthetic_row_carries_no_result_of_its_own(league_dir):
    """
    The placeholders can never reach a feature.

    A feature describes what happened before a match, so the one thing the
    synthetic row must not contribute is its own result.
    """
    assert set(STATISTIC_COLUMNS).isdisjoint(FEATURE_COLUMNS)
    assert "FTR" in STATISTIC_COLUMNS
    assert "home_xg" in STATISTIC_COLUMNS


def test_placeholders_are_blank_rather_than_zero(league_dir):
    """
    A 0-0 placeholder would be indistinguishable from a real goalless draw."""
    rows = build_rows()
    predictor = trained_predictor(league_dir, rows)

    ordered, position = predictor._fixture_frame(
        "Arsenal", "Chelsea", predictor.default_date()
    )

    row = ordered.iloc[position]

    for column in STATISTIC_COLUMNS:
        assert pd.isna(row[column]), column


def test_probabilities_are_labelled_by_class_not_by_column(league_dir):
    """
    Outcome letters come from the model, not from a column index.

    A classifier returns its classes in label-sorted order, so reading column 0
    as a fixed outcome is only right while that order stays A, D, H.
    """
    class ShuffledModel:
        """A model whose classes come back in an unhelpful order."""

        classes_ = np.array(["H", "A", "D"])

        def predict_proba(self, frame):
            return np.array([[0.2, 0.5, 0.3]])

    rows = build_rows()
    predictor = trained_predictor(league_dir, rows)

    frame = create_elo_features(
        create_features(predictor.dataframe)
    ).iloc[-1]

    probabilities = predictor._probabilities(ShuffledModel(), frame)

    assert probabilities == {"H": 0.2, "A": 0.5, "D": 0.3}


def test_predictions_are_a_distribution_over_the_three_outcomes(league_dir):
    """Probabilities are keyed by outcome and sum to one."""
    rows = build_rows()
    predictor = trained_predictor(league_dir, rows)

    prediction = predictor.predict(*unplayed_pair(rows))

    for probabilities in prediction.probabilities.values():
        assert set(probabilities) == {"H", "D", "A"}
        assert sum(probabilities.values()) == pytest.approx(1.0)


def test_the_outcome_is_the_most_likely_one(league_dir):
    """The reported outcome agrees with the reported probabilities."""
    rows = build_rows()
    predictor = trained_predictor(league_dir, rows)

    prediction = predictor.predict(*unplayed_pair(rows))

    for label, outcome in prediction.outcomes.items():
        probabilities = prediction.probabilities[label]

        assert outcome == max(probabilities, key=probabilities.get)


def test_team_names_are_matched_without_regard_to_case(league_dir):
    """'arsenal' and 'ARSENAL' both find the club."""
    rows = build_rows()
    predictor = trained_predictor(league_dir, rows)

    assert predictor.resolve_team("arsenal") == "Arsenal"
    assert predictor.resolve_team("  ARSENAL  ") == "Arsenal"


def test_an_unknown_team_is_refused_with_suggestions(league_dir):
    """A name that is not in the league fails, and points at what is."""
    rows = build_rows()
    predictor = trained_predictor(league_dir, rows)

    with pytest.raises(UnknownTeamError) as error:
        predictor.resolve_team("Livrerpool")

    assert "Liverpool" in str(error.value)


def test_a_fixture_that_has_already_been_played_is_refused(league_dir):
    """Asking about a fixture whose result is on file is not useful."""
    rows = build_rows()
    predictor = trained_predictor(league_dir, rows)

    home, away, date = pick_row(rows, 10)

    with pytest.raises(AlreadyPlayedError) as error:
        predictor.predict(home, away, date)

    assert "already been played" in str(error.value)


def test_the_next_meeting_of_two_clubs_is_still_predicted(league_dir):
    """Having met before does not make the next meeting unpredictable.

    Two clubs meeting again is the normal case, not a reason to refuse. A
    refusal here would rule out most real fixtures, since almost every league
    fixture is a repeat of a pairing from the same or a previous season.
    """
    rows = build_rows()
    predictor = trained_predictor(league_dir, rows)

    home, away, _ = pick_row(rows, 10)

    prediction = predictor.predict(home, away, "2030-05-01")

    assert prediction.home == home
    assert prediction.away == away
    assert prediction.date == pd.Timestamp("2030-05-01")


def test_a_date_the_data_has_already_reached_is_refused(league_dir):
    """A date the results are already known for is refused with the date noted.

    The fixture may not be in the data at all, but the data still extends past
    the date asked for, so nothing about it can be predicted honestly.
    """
    rows = build_rows()
    predictor = trained_predictor(league_dir, rows)

    home, away, date = pick_row(rows, 30)

    earlier = date - pd.Timedelta(days=7)

    with pytest.raises(AlreadyPlayedError) as error:
        predictor.predict(home, away, earlier)

    message = str(error.value)

    assert "after the date asked for" in message
    assert str(earlier.date()) in message


def test_a_team_cannot_play_itself(league_dir):
    """Both sides the same is a typo, not a fixture."""
    rows = build_rows()
    predictor = trained_predictor(league_dir, rows)

    with pytest.raises(PredictionError, match="cannot play itself"):
        predictor.predict("Arsenal", "Arsenal")


def test_a_new_season_regresses_the_ratings(league_dir):
    """
    A fixture past the season boundary gets regressed ratings, and says so.

    The date is not decoration. The pipeline regresses every rating towards the
    mean when the season changes, so the same fixture dated either side of the
    boundary is a different question. The direction depends on which side of
    1500 the club sits, so what is checked is the distance to the mean.
    """
    rows = build_rows()
    predictor = trained_predictor(league_dir, rows)

    home, away = unplayed_pair(rows)

    within = predictor.predict(home, away, "2025-01-15")
    beyond = predictor.predict(home, away, "2025-08-15")

    assert abs(beyond.elo["home"] - 1500) < abs(within.elo["home"] - 1500)
    assert beyond.elo["home"] == pytest.approx(
        A_FACTOR * within.elo["home"] + (1 - A_FACTOR) * 1500
    )
    assert any("regressed" in note for note in beyond.notes)
    assert not any("regressed" in note for note in within.notes)


def test_a_date_before_the_data_is_flagged(league_dir):
    """A past date means the window is that date's, not today's."""
    rows = build_rows()
    predictor = trained_predictor(league_dir, rows)

    prediction = predictor.predict(*unplayed_pair(rows), "2024-09-01")

    assert any("before the last match" in note for note in prediction.notes)


def test_a_thin_history_is_flagged(league_dir):
    """
    A newly promoted club is predicted with a warning.

    Three matches leave the rolling window nearly empty and the Elo barely off
    1500, which would otherwise be presented as an established rating.
    """
    rows = build_rows(count=12)
    rows.append({
        "Date": "01/09/2024",
        "Div": "E0",
        "HomeTeam": "Newly Promoted",
        "AwayTeam": "Arsenal",
        "FTHG": 0,
        "FTAG": 2,
        "FTR": "A",
        "HS": 5,
        "AS": 15,
        "HST": 1,
        "AST": 6,
        "home_xg": 0.4,
        "away_xg": 2.1,
    })

    predictor = trained_predictor(league_dir, rows)

    prediction = predictor.predict("Newly Promoted", "Chelsea")

    assert any(
        "Newly Promoted" in note and "1500" in note
        for note in prediction.notes
    )


def test_a_missing_xg_window_is_flagged(league_dir):
    """No measured xG in the window means the model is working without it."""
    rows = build_rows(with_xg=False)

    for row in rows:
        row["home_xg"] = None
        row["away_xg"] = None

    predictor = trained_predictor(league_dir, rows)

    prediction = predictor.predict(*unplayed_pair(rows))

    assert any("xG" in note for note in prediction.notes)


def test_the_date_defaults_to_a_week_after_the_last_match(league_dir):
    """A fixture asked for without a date sits a week on."""
    rows = build_rows()
    predictor = trained_predictor(league_dir, rows)

    prediction = predictor.predict(*unplayed_pair(rows))

    assert prediction.date == predictor.data_through + pd.Timedelta(days=7)


def test_the_form_figures_come_from_the_feature_vector(league_dir):
    """
    The form line and the features cannot disagree.

    The summary is read out of the same vector the model is given, so there is
    no second calculation to drift.
    """
    rows = build_rows()
    predictor = trained_predictor(league_dir, rows)

    prediction = predictor.predict(*unplayed_pair(rows))

    assert prediction.form["home"]["pts"] == pytest.approx(
        float(prediction.features["HomePT5"])
    )
    assert prediction.form["away"]["xG"] == pytest.approx(
        float(prediction.features["AwayXG5"])
    )


def test_league_names_are_accepted_in_their_common_spellings():
    """EPL, LaLiga and Serie A all find their league."""
    assert resolve_league("EPL") == "PremierLeague"
    assert resolve_league("epl") == "PremierLeague"
    assert resolve_league("PremierLeague") == "PremierLeague"
    assert resolve_league("Laliga") == "LaLiga"
    assert resolve_league("Serie A") == "SerieA"


def test_an_unknown_league_is_refused():
    """There is no default league to fall back on."""
    with pytest.raises(UnknownLeagueError):
        resolve_league("Bundesliga")


def test_the_rendered_block_shows_both_models_and_the_elo(league_dir):
    """The output a person reads carries the numbers it claims to."""
    rows = build_rows()
    predictor = trained_predictor(league_dir, rows)

    rendered = format_prediction(predictor.predict(*unplayed_pair(rows)))

    assert "Random Forest" in rendered
    assert "XGBoost" in rendered
    assert "Home" in rendered and "Draw" in rendered and "Away" in rendered
    assert "Elo" in rendered
    assert "Form, last 5" in rendered


def test_the_rendered_block_can_narrow_to_one_model(league_dir):
    """--model narrows what is shown without changing what was computed."""
    rows = build_rows()
    predictor = trained_predictor(league_dir, rows)

    rendered = format_prediction(
        predictor.predict(*unplayed_pair(rows)),
        models=["XGBoost"],
    )

    assert "XGBoost" in rendered
    assert "Random Forest" not in rendered


def test_explain_prints_every_feature(league_dir):
    """--explain shows the whole vector, so a surprise can be traced."""
    rows = build_rows()
    predictor = trained_predictor(league_dir, rows)

    rendered = format_prediction(
        predictor.predict(*unplayed_pair(rows)), explain=True
    )

    for column in FEATURE_COLUMNS:
        assert column in rendered
