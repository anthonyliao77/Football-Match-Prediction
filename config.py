"""
Stores feature definitions for the evaluation model.
"""

TRAIN_END_DATE = "2025-05-31"
VALIDATION_START_DATE = "2025-08-15"
VALIDATION_END_DATE = "2026-06-01"

feature_columns = [
    # Team points last five matches
    "HomePT5",
    "AwayPT5",
    # Team goals scored last five matches
    "HomeGS5",
    "AwayGS5",
    # Team goals conceded last five matches
    "HomeGC5",
    "AwayGC5",
    # Team goal difference last five matches
    "HomeGD5",
    "AwayGD5"
]
