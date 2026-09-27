<sub>Part of [Football Match Prediction](https://github.com/anthonyliao77/Football-Match-Prediction). Start at the [README](https://github.com/anthonyliao77/Football-Match-Prediction#readme).</sub>

# API-Football (optional)

== API-Football Credentials


Training works entirely from the local CSV files and needs no credentials. To
augment those files with API-Football data, create a `.env` file in the project
root:

```bash
API_FOOTBALL_KEY=your_api_key_here
```

The key is read only when an API request is made, so the project still imports
and trains normally without it. `.env` is listed in `.gitignore` and must not be
committed.

Get a key from [api-football.com](https://www.api-football.com/). The free plan
is enough to try the tooling, but it restricts which seasons can be queried and
does not allow the `ids` parameter, so match statistics cannot be fetched with
it. A paid plan removes both limits.

== Commands


Preview a change without writing anything:

```bash
python update_data.py --league PremierLeague --season 2026 --dry-run
```

Note that `--dry-run` still makes the API calls, so it does spend quota. It
suppresses the file write, not the request.

Fetch recently completed fixtures, restricted to a date window to keep the
request count down:

```bash
python update_data.py --league PremierLeague --season 2026 \
    --from 2026-09-19 --to 2026-09-26
```

Backfill a whole season from the fixture schedule:

```bash
python update_data.py --league PremierLeague --season 2026 --schedule
```

== Request cost


A daily window covering a handful of matches costs 1 request for the fixtures
plus 1 per 20 matches for statistics. A full season with no window costs about
20. Restricting `--from` and `--to` is the single biggest lever on quota usage.

== Statistics availability


The `/fixtures` endpoint does not return match statistics inline, so shots and
shots on target are fetched in a second call that passes fixture IDs. Plans
without access to the `ids` parameter cannot make that call. When it is
rejected, the update still ingests scores and results, prints a warning, and
leaves the shots columns untouched.

== How the merge behaves


API rows are matched to existing CSV rows on **date, home team and away team**:

* A **matched** row has only its *empty* cells filled in. Existing
  football-data.co.uk values always win, so scores and the odds and handicap
  columns are never overwritten.
* An **unmatched** row is appended, but only if it has a final score.
* Running the same update twice leaves the file unchanged.

Team names differ between the two sources, so they are translated through
`API_FOOTBALL_TEAM_MAP` in `config.py`. An API team name that is neither
identical to the CSV spelling nor listed there is **rejected** with an error,
because ingesting it would file the club under a second name and split its Elo
rating and rolling form in two. When a team is promoted or renamed, add the
mapping before running an update.

`Referee` is written for the Premier League only, which is the only set of
CSVs that has the column. The value is not read anywhere in the model.

== What a free API-Football key can and cannot do


The key this project was developed against is on the Free plan, and that plan
is genuinely limited. Measured against it, not assumed:

| Capability | Free plan |
| --- | --- |
| Request rate | 10 requests per minute |
| Daily requests | 100 |
| Seasons reachable | 2022/2023 to 2024/2025 |
| Current 2026/2027 season | No |
| Batched `ids` lookups | No |

Consequences worth planning around:

* **The current season cannot be ingested via the API at all.** A 2026/2027
  update returns nothing useful on this plan, so current-season data comes from
  the football-data.co.uk CSVs and `update_data.py` is useful for the older
  seasons the plan can still reach.
* **The statistics call fails**, because it needs the `ids` parameter. This is
  why shots come from football-data rather than the API, and why xG comes from
  Understat directly rather than from the API.
* **`--dry-run` still spends quota.** It suppresses the write, not the request.

Anyone with a paid key gets working current-season updates, and API-Football
also serves an `expected_goals` statistic that could serve as a second opinion
on xG. That path is **not** implemented: it cannot be tested against the free
key, and shipping an unvalidated xG source would be worse than the measured
Understat values the project already has.
