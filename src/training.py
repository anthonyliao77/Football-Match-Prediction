"""
Trains, validates and evaluates the prediction model.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    # confusion_matrix,
    log_loss,
)
from sklearn.preprocessing import LabelEncoder

# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier

from config import (
    FEATURE_COLUMNS,
    LEAGUES,
)
from src.data_loader import load_data, split_by_season
from src.elo import create_elo_features, get_season
from src.features import create_features
from src.xg import XG_COLUMNS


def _xg_coverage_by_season(dataframe) -> pd.DataFrame:
    """
    Counts, per season, how many rows carry measured xG and how many do not.

    Parameters:
        dataframe: DataFrame containing match data with xG columns.

    Returns:
        A frame indexed by season with a row count and a gap count.
    """
    seasons = dataframe["Date"].map(get_season)
    gaps = dataframe[XG_COLUMNS].isna().any(axis=1)

    return (
        pd.DataFrame({"season": seasons, "gap": gaps})
        .groupby("season")["gap"]
        .agg(rows="size", gaps="sum")
    )


def _warn_on_missing_xg(dataframe, league: str) -> None:
    """
    Reports the xG the run is about to do without.

    xG reaches the model only through four rolling features, and a row with no
    xG contributes zero to them. That is invisible in the output, so it is
    called out here: a validation season without xG produces scores that look
    worse than the model is, and nothing in the metrics would say why.

    Parameters:
        dataframe: DataFrame containing match data with xG columns.
        league (str): The league name, for the suggested command.
    """
    coverage = _xg_coverage_by_season(dataframe)

    measured = int(coverage["rows"].sum() - coverage["gaps"].sum())

    print(
        f"xG coverage: {measured} of {len(dataframe)} rows carry measured xG"
    )

    # split_by_season holds out the most recent season, so that is the one the
    # reported scores are computed on.
    newest = coverage.index[-1]
    gaps = int(coverage.loc[newest, "gaps"])
    rows = int(coverage.loc[newest, "rows"])

    if not gaps:
        return

    print(
        f"\n{'!' * 72}\n"
        f"WARNING: {gaps} of {rows} rows in {newest} have no measured xG.\n"
        f"Understat has not published that season, so HomeXG5, AwayXG5,\n"
        f"HomeXGA5 and AwayXGA5 will read 0 for every row of the validation\n"
        f"set. The scores below are therefore pessimistic, and the cause will\n"
        f"not be visible in the metrics themselves.\n"
        f"Try: python backfill_xg.py --league {league}\n"
        f"{'!' * 72}"
    )


def train_model(league):
    """
    Train a model for the specified league using
    historical match data and features.

    Args:
        league (str): The name of the league to train the model on. Must be one
        of the keys in the LEAGUES dictionary.

    Raises:
        ValueError: If the specified league is not found in the LEAGUES
        dictionary.
    """
    if league not in LEAGUES:
        raise ValueError(
            f"Unknown league '{league}'. "
            f"Available leagues: {list(LEAGUES)}"
        )

    league_config = LEAGUES[league]

    # Load and combine match data for the specified league
    dataframe = load_data(
        league=league_config["football_data"]
    )

    # Checked before the features are built, because create_features returns a
    # fresh frame and does not carry the xG columns through.
    _warn_on_missing_xg(dataframe, league)

    # Create features for the model using the combined match data
    dataframe = create_features(dataframe=dataframe)
    dataframe = create_elo_features(dataframe=dataframe)

    # Split the data into training and validation sets based on date ranges
    train_data, validation_data = split_by_season(
        dataframe,
        validation_seasons=1
    )

    # training data
    train_X = train_data[FEATURE_COLUMNS]
    train_y = train_data["FTR"]

    # validation data
    val_X = validation_data[FEATURE_COLUMNS]
    val_y = validation_data["FTR"]

    # Encode target labels for XGBoost
    # Convert A/D/H to 0/1/2 for multiclass classification
    label_encoder = LabelEncoder()

    train_y_encoded = label_encoder.fit_transform(train_y)
    val_y_encoded = label_encoder.transform(val_y)

    # Random Forest model
    random_forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=67,
        n_jobs=-1
    )

    random_forest.fit(train_X, train_y)

    # Random Forest prediction
    prediction = random_forest.predict(val_X)
    rf_probabilities = random_forest.predict_proba(val_X)

    rf_results = validation_data[
        ["Date", "HomeTeam", "AwayTeam", "FTR"]
    ].copy()

    rf_results["Prediction"] = prediction

    # Add predicted probabilities to the results DataFrame
    rf_results["AwayProbability"] = rf_probabilities[:, 0]
    rf_results["DrawProbability"] = rf_probabilities[:, 1]
    rf_results["HomeProbability"] = rf_probabilities[:, 2]

    # # Random Forest accuracy
    rf_accuracy = accuracy_score(val_y, prediction)

    # Log loss
    rf_log_loss = log_loss(
        val_y,
        rf_probabilities,
        labels=random_forest.classes_
    )

    # Multiclass Brier score
    one_hot_y = pd.get_dummies(
        val_y
    ).reindex(
        columns=random_forest.classes_,
        fill_value=0
    )

    rf_brier_score = (
        (rf_probabilities - one_hot_y.to_numpy()) ** 2
    ).sum(axis=1).mean()

    print(
        rf_results[
            [
                "Date",
                "HomeTeam",
                "AwayTeam",
                "FTR",
                "Prediction",
                "HomeProbability",
                "DrawProbability",
                "AwayProbability",
            ]
        ].head(10)
    )
    print(
        "Random Forest accuracy:", rf_accuracy,
        "Log loss:", rf_log_loss,
        "Brier score:", rf_brier_score,
        "\n"
    )

    # XGBoost model
    xgb = XGBClassifier(
        n_estimators=200,
        random_state=67,
        learning_rate=0.01,
        n_jobs=-1
    )

    xgb.fit(train_X, train_y_encoded)

    # XGBoost prediction
    xgb_prediction_encoded = xgb.predict(val_X)
    xgb_probabilities = xgb.predict_proba(val_X)

    # Convert predictions back to A/D/H
    xgb_prediction = label_encoder.inverse_transform(
        xgb_prediction_encoded
    )

    # XGBoost results
    xgb_results = validation_data[
        ["Date", "HomeTeam", "AwayTeam", "FTR"]
    ].copy()

    xgb_results["Prediction"] = xgb_prediction

    # Add probabilities
    xgb_results["AwayProbability"] = xgb_probabilities[:, 0]
    xgb_results["DrawProbability"] = xgb_probabilities[:, 1]
    xgb_results["HomeProbability"] = xgb_probabilities[:, 2]

    # XGBoost accuracy
    xgb_accuracy = accuracy_score(
        val_y,
        xgb_prediction
    )

    # Log loss
    xgb_log_loss = log_loss(
        val_y_encoded,
        xgb_probabilities,
        labels=range(len(label_encoder.classes_))
    )

    # Multiclass Brier score
    one_hot_y = pd.get_dummies(
        val_y_encoded
    ).reindex(
        columns=range(xgb_probabilities.shape[1]),
        fill_value=0
    )

    xgb_brier_score = (
        (xgb_probabilities - one_hot_y.to_numpy()) ** 2
    ).sum(axis=1).mean()

    # Print results
    print(
        xgb_results[
            [
                "Date",
                "HomeTeam",
                "AwayTeam",
                "FTR",
                "Prediction",
                "HomeProbability",
                "DrawProbability",
                "AwayProbability",
            ]
        ].head(10)
    )

    print(
        "XGBoost accuracy:", xgb_accuracy,
        "Log loss:", xgb_log_loss,
        "Brier score:", xgb_brier_score
    )

    # # Random Forest confusion matrix
    # print(random_forest.classes_)
    # print(confusion_matrix(val_y, prediction))

    # # XGBoost confusion matrix
    # print(label_encoder.classes_)
    # print(
    #     confusion_matrix(
    #         val_y,
    #         xgb_prediction,
    #         labels=label_encoder.classes_
    #     )
    # )

    # # Match distribution of validation data
    # print(validation_data["FTR"].value_counts(normalize=True))

    # # Random Forest feature importance
    # rf_importance = pd.Series(
    #     random_forest.feature_importances_,
    #     index=train_X.columns
    # ).sort_values(ascending=False)

    # print("Random Forest feature importance:", rf_importance)

    # # XGBoost feature importance
    # xgb_importance = pd.Series(
    #     xgb.feature_importances_,
    #     index=train_X.columns
    # ).sort_values(ascending=False)

    # print("XGBoost feature importance:", xgb_importance)
