"""
The columns that carry results and shot counts in the season CSVs.

Kept apart from the code that fills them for the same reason src/xg.py is: the
script that writes these values and the pipeline that reads them need to agree on
the names without either depending on the other.

The result columns and the shot columns are listed separately because they are
not filled from the same place. A score comes from the league endpoint, which
the refresh already reads, and costs nothing extra to take. A shot count does
not: Understat's league payload has no shots on it at all, so those values cost
one request per match, and only the ones already on file from
football-data.co.uk are left alone.
"""

# The score, three columns, which together are what makes a row a match rather
# than a fixture.
RESULT_COLUMNS = ["FTHG", "FTAG", "FTR"]

# The shots, four columns: total and on target for each side.
SHOT_COLUMNS = ["HS", "AS", "HST", "AST"]

# Everything above, which is what a match filled from Understat has to have
# before it is worth writing.
FILLED_COLUMNS = RESULT_COLUMNS + SHOT_COLUMNS

# Records where a row's result came from, so a value fetched from Understat can
# be told apart from one that arrived with the football-data.co.uk download and
# from one somebody typed in.
RESULT_SOURCE_COLUMN = "result_source"

# Written to RESULT_SOURCE_COLUMN for a result read from Understat.
UNDERSTAT_RESULT_SOURCE = "understat"

# Written to RESULT_SOURCE_COLUMN for a result that was already on file. Nothing
# writes it: a row that arrives with the download is understood to be from the
# download, and a column blank on every historical row is a column that cannot
# be relied on to mean anything.
FOOTBALL_DATA_RESULT_SOURCE = "football-data"
