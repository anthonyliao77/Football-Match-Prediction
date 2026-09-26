"""
The columns that carry expected-goals data in the season CSVs.

Kept apart from the code that fills them so that the backfill script and the
training pipeline can agree on the names without either depending on the other.
"""

# The two xG values, one per side of a fixture.
XG_COLUMNS = ["home_xg", "away_xg"]

# Records where a row's xG came from, so a value that was entered by hand can be
# told apart from one that was fetched.
SOURCE_COLUMN = "xg_source"

# The value written to SOURCE_COLUMN for xG read from Understat.
UNDERSTAT_SOURCE = "understat"
