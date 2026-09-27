<sub>Part of [Football Match Prediction](https://github.com/anthonyliao77/Football-Match-Prediction). Start at the [README](https://github.com/anthonyliao77/Football-Match-Prediction#readme).</sub>

# Data

== Football-Data


Historical match statistics are stored locally in:

```text
football_data/
├── LaLiga/
├── PremierLeague/
└── SerieA/
```

The data provides information such as:

* Match results
* Goals
* Shots
* Shots on target
* Other historical match statistics

The data is loaded, combined, normalized, and sorted chronologically before feature engineering.

No API key is required for these local files. They can optionally be augmented with API-Football fixtures, which does require a key (see [Updating Match Data](#updating-match-data)).

== Understat


Understat provides historical expected-goals information.

There is no public Understat API, so `src/understat_client.py` reads the same
JSON endpoint the website itself calls:

```text
https://understat.com/getLeagueData/{league}/{start_year}
```

where `{league}` is one of `EPL`, `La_liga` and `Serie_A`, and `{start_year}` is
the first year of the season, so 2026/2027 is requested as `2026`. The response
lists every fixture of that season with both xG values, which are then matched
against the local CSVs.

Two details are load-bearing:

* **The endpoint needs a session.** It answers 404 until the client has first
  visited the site and holds its cookies, so the client fetches the homepage
  once before the first request. A 404 caused by a missing cookie is
  indistinguishable from a season that does not exist, which is a very
  confusing way to lose a season.
* **One request per season.** The endpoint can also be asked for several
  seasons at once, but it is answered as a unit: if the newest season is not
  published, the whole response comes back short and the missing data is only
  visible by counting rows. Requesting each season separately means an
  unpublished season costs exactly that season.

Only fixtures the site marks as played are kept. A fixture that has not been
played carries `xG: {h: null, a: null}`, and a season with no played matches at
all is reported as unavailable rather than returned as an empty frame.

Requests are sequential with a small delay and there is no cache. That is
deliberate: a cached in-progress season would go stale and silently, which is
the one failure mode that would be hardest to notice.

== A season CSV holds the whole season, played and upcoming


Each season file holds that season's full fixture list, not just the results.
A fixture that has not been played yet is present with a date and two clubs,
and blank everywhere else: result, goals, shots and xG. This is what makes the
season being predicted known in advance, and it is why the whole file is 380
rows rather than the handful played so far.

Those blank rows are inert. Everything downstream filters on a result being
present, so an unplayed fixture never reaches the rolling form, the xG columns
or the Elo pass. `test_a_pending_fixture_in_the_csv_cannot_change_a_prediction`
holds that line: padding a season file with the rest of the schedule cannot move
a prediction.

Two consequences for anyone reading the files by hand:

* A blank `FTR` means **not yet played**, not a nil-nil draw. A nil-nil draw has
  `FTR` of `D`.
* Row count is no longer a proxy for how much data a season has. The number of
  played matches is.

== Adding the schedule and filling in results


`sync_understat.py` keeps a season CSV in step with Understat: it adds the
fixtures still to come, and it writes the results of matches that have since been
played. Understat lists a full season before kickoff, and publishes a score as
soon as a match ends, so one script covers both ends of a season's life:

```bash
python sync_understat.py --league PremierLeague --season 2026/2027
```

Use `--dry-run` first to see what would change, and `--season` to target a
specific season. The rules it works by:

* It **only adds or fills blanks**. A cell that already holds a value is left
  exactly as it is, so every result downloaded from football-data.co.uk is
  never overwritten. Understat fills the gaps the download has not reached yet,
  which is why the two sources coexist and neither one has to be re-downloaded.
* A match the two sources **disagree** on is left alone and reported. The value
  on file wins, and the difference is printed rather than resolved, because
  which of them is right is not something a script can decide.
* A row that already carries a result but is **missing shot counts** is left
  alone and reported, by the same reasoning: the goals on it came from
  football-data.co.uk and the shots would come from Understat, and mixing the two
  inside one match is worse than an honest gap. The all-or-nothing rule below
  protects the rows this script writes; this is the same corrupt shape arriving
  by another route, and it is named rather than tolerated.
* It matches on date, home team and away team, and translates Understat's club
  names through `TEAM_NAME_MAP` first.
* It writes only the columns the file already has, so a season that has never
  been given an xG column does not acquire an empty one.
* A club the season file has never seen is reported and left out, because two
  spellings of one club would split its Elo rating and its form in two.
* A **postponed** fixture has its date rewritten in place rather than added
  again. A league does play the same pairing more than once in a season, so only
  rows with no result are eligible to be re-dated; by the time the return leg is
  listed, the first one has been played and is left alone.
* A played match the file has never heard of is **added**, decided, with its
  shots. More often it is a fixture added before kickoff and still sitting there
  blank.

== Keeping the data current


```bash
python refresh_data.py --dry-run   # report
python refresh_data.py             # write
```

This runs both Understat-backed steps for every league, fixtures then xG, and
prints one table:

```text
League            Fixtures  Re-dated  Results      xG  Unwritten
----------------------------------------------------------------
PremierLeague            0         0        0       0         0
LaLiga                   0         0        0       0         0
SerieA                   0         0        0       0         0
```

`--league` narrows it to one league and `--season` to one season of fixtures.
The xG step always covers the whole league, since it only fills cells that are
empty and narrowing it would hide gaps in older seasons.

An **Unwritten** count is the number of played matches the CSV has no result for.
That is the one figure worth reading, and it is the reason the command exists.
It is the only place in the project that can say a played match is missing,
because a fixture added before kickoff sits in the file with a blank result,
matches on date and sides, and is counted as *already present* by both
underlying scripts. Nothing else says so. Meanwhile a blank row reads as a match
that has not happened, so it is dropped from the rolling features and from the
Elo, and a club's recent form goes stale quietly.

**Results come from Understat too.** A blank result is no longer a chore left
undone, and the refresh no longer tells you to download anything. Understat
publishes a score as soon as a match ends, so the file fills itself in now, and
a non-zero Unwritten count means the refresh could not *finish*: an abandoned or
unplayed fixture, a match whose shot counts could not be read, or a request that
did not come back. Each one is listed by name by the fixtures step.

The exit status is non-zero if any league could not be refreshed, and a league
that fails does not stop the others. Understat publishes no API and no
availability, so treat a failure as retryable rather than as a broken dataset.

A league that failed shows a **dash** in the Unwritten column rather than a
zero, because the refresh died before it could count. A dash is not a zero, and
the run says so instead of reporting the leagues it did manage to check as proof
that everything is current.

== Before a prediction


```bash
python refresh_data.py --require-fresh
```

The default exit status catches a league failing, which is the loud failure. It
does not catch the likelier one: both Understat steps succeed, the tables are
clean, and one match is still sitting unwritten, so the run is green and the
model trains on a season that is quietly short a game. `--require-fresh`
extends the non-zero exit to a non-zero Unwritten count, and to any league whose
count is unknown.

Gate a prediction on it, and chain the two so the prediction cannot run on a
season that is short a game:

```cron
17 7 * * *  cd /path/to/repo && .venv/bin/python refresh_data.py --require-fresh \
             && .venv/bin/python predict.py --league PremierLeague --home Arsenal --away Leeds
```

A `--require-fresh` run will start failing the moment a matchday ends and the
next refresh has not yet run. That is the correct behaviour here: it is telling
you the model is about to be trained on results that are not there yet.

== As a monitor


The same command is the wrong guard for a check that runs on a timer, for a
reason that has nothing to do with correctness. Committed data goes stale after
every matchday and only becomes current once the refresh is run and its output
committed. Asked daily, a strict check is red most of the time, and a red check
people have learned to ignore is worse than no check.

`--max-age-days N` narrows the failure to a match that has been sitting
unwritten for more than N days:

```bash
python refresh_data.py --dry-run --require-fresh --max-age-days 7
```

A match blank since this morning is one nobody has got to yet. One blank for
three weeks is a fault nobody has looked at. The table shows both counts, so a
clean exit next to a non-zero Unwritten is visible rather than confusing:

```text
League            Fixtures  Re-dated  Results      xG  Unwritten  Overdue
----------------------------------------------------------------
PremierLeague            0         0        0       0          3        2
```

Two things it does not excuse. Without the flag the behaviour is exactly as it
was, since `0` would mean "older than today" and would quietly stop counting a
match played a few hours ago. And a league that *failed* still fails either way,
because a league nobody could count is not a young match — the threshold filters
match ages, and it cannot turn unknown into fine.

Ages are measured against UTC, so a local run and a scheduled one agree about
what "three days old" means.

== The two scheduled workflows


Both run daily and both can be started by hand from the Actions tab.

| Workflow               | Time (UTC) | Permission       | What it does                                                       |
| ---------------------- | ---------- | ---------------- | ------------------------------------------------------------------ |
| `data-freshness.yaml`  | 07:17      | `contents: read` | Dry-runs the refresh with a 7-day grace period, and fails if a match has been unwritten longer than that |
| `data-refresh.yaml`    | 08:17      | `contents: write`| Refreshes for real and commits the diff to `main`                  |

The check runs first and reads yesterday's settled tree. What it is for is the
failure the refresh cannot fix by itself: a match abandoned at Understat, a
league unreachable for a week, a push rejected because the branch moved. Those
are the cases where data is still unwritten the next morning despite a refresh
having run, and a check that only ever agreed with the writer would never notice
them.

**The refresh commits to `main` directly, with no pull request.** The diff is
only ever season CSVs, and the values in them are the same Understat figures the
project already trusts, so there is nothing in it for a human to decide. Every
committed value carries its origin in the `result_source` column, so a row that
was not on file before is still traceable after it lands.

`data-refresh.yaml` holds `contents: write`, which is the one genuinely widened
permission here, and it is what the guard is for: if anything other than a
season CSV is ever staged, the job refuses to commit and reports what it found.
The refresh cannot write code into the repository, so the realistic worst case
is bad numbers in `football_data/` rather than a backdoor.

A league that cannot be read does not stop the job at the point it fails. Each
league is written all-or-nothing, so the ones that did read cleanly are still
committed, and the job goes red afterwards so the failure is not mistaken for a
quiet day. The push is neither forced nor rebased: a rejected push means someone
landed on `main` mid-run, and the next run reconciles against it rather than
overwriting it.
