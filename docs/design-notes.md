<sub>Part of [Football Match Prediction](https://github.com/anthonyliao77/Football-Match-Prediction). Start at the [README](https://github.com/anthonyliao77/Football-Match-Prediction#readme).</sub>

# Design notes

== Why the numbers can be trusted


The hard part is not fitting a model but making sure the feature vector a
prediction is scored against is the one the model was fitted on. Recomputing
features for a fixture by hand is where this normally goes wrong, so nothing is
recomputed.

Instead the fixture is appended to the data as a synthetic row with its result,
score and xG left empty, and the ordinary `create_features` and
`create_elo_features` pass runs over the result. The synthetic row's own features
are then read back by position.

```text
Real matches ──► append one synthetic row ──► the normal feature pass
                                                      │
                                     read that row's features by position
```

That is the same code path training uses, so the two cannot drift apart. The
rolling statistics naturally read the five matches before the fixture, and the
Elo rating is whatever it was immediately before kickoff, with new-season
regression applied if the date crosses the July boundary.

The test suite checks this directly: one match is deleted from the data, asked
for by name, and every one of its twenty features is asserted equal to the
vector the training pipeline produced for that row while it was still there. The
row is then also checked not to be the last row in the frame, because taking the
last row would be the natural shortcut and would return a different match's
features.

The result, score and xG columns are left empty rather than filled with zero on
purpose. A placeholder `0-0` would be indistinguishable from a real goalless
draw if the row's position ever shifted.

== Cost and honesty about it


There is no saved model. Every run loads the CSVs, rebuilds the features and
Elo ratings, and refits both classifiers on every season. Measured on the
Premier League data, that is about 1.8s to fit and about 0.5s per fixture
afterwards, so roughly 2.7s for a single command-line prediction. That is a
deliberate trade for a codebase with no artifact to go stale, and it is the
thing to revisit first if this ever needs to be quick.

The model is fitted on every season **before** the validation season, and the
season being predicted is in neither the fit nor the validation split. A season
after the validation season is not quietly folded into training: fitting on it
would train the model on the very matches it is about to predict, and the
reported scores would stop describing the model being used.

The rolling features and Elo ratings behind a prediction are a separate
question from the fit. Those are built from every match played so far,
including the season being predicted, because a result that has already happened
is real information about how a team is playing. A fixture is only ever a date
and two clubs, so a fixture that has not been played is dropped before the
feature and Elo passes: those passes are chronological and would read a row
with no result as a zero for both sides, spending a slot in every rolling window
that follows it.

Caveats travel with the prediction rather than being buried here. A team with a
thin history, a date older than the data, a season boundary and an unmeasured
xG window are each reported in the output, because a probability printed
without them invites reading more into it than it can carry.

== Reproducibility


For comparable results, the following should remain consistent:

* Historical CSV data
* Understat data
* Feature configuration
* Team-name mappings
* Elo parameters
* Random seeds
* Validation season
* Python/library versions

The current model configurations use:

```text
Random Forest random_state = 67
XGBoost random_state       = 67
```

The validation process is deterministic with respect to the same input data and configuration.

For stronger reproducibility in the future, dependency versions and experiment configurations should be pinned.

== Data Leakage Considerations


Avoiding future information is a central consideration in this project.

The current pipeline attempts to prevent common forms of leakage by:

* Keeping matches in chronological order
* Splitting validation by season
* Using only previous matches for rolling features
* Calculating Elo sequentially
* Using pre-match Elo values as model features
* Updating Elo only after processing the current result
* Holding out a later season rather than randomly sampling matches from the same period

For example, when predicting:

```text
Team A vs Team B
```

the model should only receive information that would have been available immediately before that fixture.

The project does not claim to be completely leakage-free. Dataset alignment, feature engineering, and future changes to the pipeline should continue to be reviewed for temporal leakage.

== Future Work


Planned improvements include:
