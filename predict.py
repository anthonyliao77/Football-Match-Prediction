"""
Predicts the outcome of a football fixture that has not been played yet.

Run with no arguments for an interactive session, or pass a fixture to predict
it in one go:

    python predict.py --league PremierLeague --home Arsenal --away Chelsea

The league is always asked for. The three leagues are not interchangeable and
defaulting to one would quietly answer a different question than the one asked.
"""

import argparse

from src.predict import (
    LEAGUES,
    MODEL_LABELS,
    AlreadyPlayedError,
    PredictionError,
    Predictor,
    UnknownLeagueError,
    UnknownTeamError,
    format_prediction,
    resolve_league,
)

EXIT_WORDS = ("q", "quit", "exit")


def parse_arguments(argv=None):
    """
    Parse command-line arguments for the prediction script.

    Parameters:
        argv (list): The arguments to read, or None to read the real ones.
    """
    parser = argparse.ArgumentParser(
        description="Predict a fixture that has not been played yet."
    )

    parser.add_argument(
        "--league",
        help="League to predict in. One of: "
        f"{', '.join(LEAGUES)}. Also accepts EPL, LaLiga and Serie A."
    )

    parser.add_argument(
        "--home",
        help="Home team, as spelled in the season CSVs."
    )

    parser.add_argument(
        "--away",
        help="Away team, as spelled in the season CSVs."
    )

    parser.add_argument(
        "--date",
        help="Date the fixture is played, as YYYY-MM-DD. Defaults to a week "
        "after the most recent match in the data."
    )

    parser.add_argument(
        "--model",
        choices=["both", "rf", "xgb"],
        default="both",
        help="Which model's prediction to show. Defaults to both."
    )

    parser.add_argument(
        "--explain",
        action="store_true",
        help="Also print the full feature vector behind the prediction."
    )

    return parser.parse_args(argv)


def selected_models(choice: str) -> list:
    """
    Turns the --model flag into the labels to display.

    Parameters:
        choice (str): "both", "rf" or "xgb".

    Returns:
        list: The model labels to show.
    """
    if choice == "both":
        return list(MODEL_LABELS.values())

    return [MODEL_LABELS[choice]]


def _ask(prompt: str) -> str:
    """
    Reads one line of input.

    Parameters:
        prompt (str): The question to put to the user.

    Returns:
        str: What the user typed, stripped, or an exit word if the input ended
            or the user interrupted it.
    """
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        # The end of the input, or Ctrl-C or Ctrl-D, means the same thing as
        # typing an exit word: stop. Letting it carry on would raise a
        # traceback at whoever was using the tool, and asking again would
        # block on input that will never come.
        print()

        return "quit"


def ask_until(predicate, question: str, error: str) -> str:
    """
    Asks a question until the answer is accepted.

    A bad answer re-asks rather than exiting, because the usual reason for one
    is a spelling the data uses differently, and the closest matches are
    usually what the person meant.

    An exit word short-circuits the check and is handed straight back, so the
    caller can leave. Testing it here rather than in each predicate matters:
    a predicate that rejected exit words would make them look like a bad
    answer, and the session would re-ask forever instead of stopping.

    Parameters:
        predicate (callable): Returns True for an acceptable answer.
        question (str): The question to put to the user.
        error (str): The message shown after a rejected answer.

    Returns:
        str: The accepted answer, or the exit word if one was typed.

    """
    while True:
        answer = _ask(question)

        if answer.lower() in EXIT_WORDS:
            return answer

        if predicate(answer):
            return answer

        print(error)


def run_interactive() -> None:
    """
    Runs a session that predicts fixtures until the user stops.

    A league's models are built the first time that league is asked for and then
    kept, so a session that stays in one league pays for training once. The
    models are held per league rather than globally because the two leagues in
    a session have nothing to share.
    """
    predictors = {}

    print(
        "Predict an unplayed fixture. Enter 'quit' at any prompt to stop.\n"
    )

    while True:
        league_name = ask_until(
            _is_league,
            f"League ({' / '.join(LEAGUES)}): ",
            "  Choose one of: " + ", ".join(LEAGUES) + ".",
        )

        if league_name.lower() in EXIT_WORDS:
            return

        league = resolve_league(league_name)

        if league not in predictors:
            print(f"\nLoading {league} and training... ", end="", flush=True)
            predictors[league] = Predictor(league)
            print("done.")

        predictor = predictors[league]

        default = predictor.default_date()

        # The predictor is bound as a default argument rather than closed over.
        # ask_until tests the predicate inside the same pass of the loop, so it
        # would behave correctly either way, but a lambda reaching for a loop
        # variable is the kind of thing that breaks the moment the loop is
        # rearranged.
        def is_known_team(answer, league_predictor=predictor) -> bool:
            """
            Reports whether a name is a team in this league, saying so if not.
            """
            return _is_team(league_predictor, answer)

        typed_home = ask_until(
            is_known_team,
            "Home team: ",
            "",
        )

        if typed_home.lower() in EXIT_WORDS:
            return

        typed_away = ask_until(
            is_known_team,
            "Away team: ",
            "",
        )

        if typed_away.lower() in EXIT_WORDS:
            return

        # Take the data's spelling, not the one that was typed. What was typed
        # has been checked and cannot fail, and using it as it stands would
        # report a team called "chelsea" as having no matches at all, because
        # the form and Elo lookups compare against the CSV spelling.
        home = predictor.resolve_team(typed_home)
        away = predictor.resolve_team(typed_away)

        fixture_date = _ask(
            f"Date (YYYY-MM-DD, blank for {default:%Y-%m-%d}): "
        )

        if fixture_date.lower() in EXIT_WORDS:
            return

        try:
            prediction = predictor.predict(home, away, fixture_date or None)
        except PredictionError as error:
            print(f"\n  {error}\n")
            continue

        print()
        print(format_prediction(prediction))

        if not ask_until(
            lambda answer: answer.lower() in ("y", "yes", "n", "no", ""),
            "\nAnother fixture? [y/N]: ",
            "  Answer y or n.",
        ).lower().startswith("y"):
            return


def _is_league(answer: str) -> bool:
    """
    Reports whether a name is a league this project covers.

    Parameters:
        answer (str): What the user typed.

    Returns:
        bool: True if the name resolves to a league.
    """
    try:
        resolve_league(answer)
    except UnknownLeagueError:
        return False

    return True


def _is_team(predictor: Predictor, answer: str) -> bool:
    """
    Reports whether a name is a team in the league, and says so if it is not.

    Parameters:
        predictor (Predictor): The league being predicted in.
        answer (str): What the user typed.

    Returns:
        bool: True if the name resolves to a team.
    """
    try:
        predictor.resolve_team(answer)
    except UnknownTeamError as error:
        print(f"  {error}")
        return False

    return True


def main(argv=None) -> None:
    """
    Parses the arguments and either predicts one fixture or starts a session.

    Parameters:
        argv (list): The arguments to read, or None to read the real ones.
            The parameter is here so a test can drive the command line without
            reaching into sys.argv and rearranging the process to suit.
    """
    args = parse_arguments(argv)

    if args.league and args.home and args.away:
        _predict_once(args)
        return

    if any((args.league, args.home, args.away, args.date)):
        raise SystemExit(
            "--league, --home and --away have to be given together. "
            "Run with no arguments for an interactive session."
        )

    run_interactive()


def _predict_once(args) -> None:
    """
    Predicts a single fixture named on the command line.

    Parameters:
        args (argparse.Namespace): The parsed arguments.
    """
    try:
        league = resolve_league(args.league)
        predictor = Predictor(league)
        home = predictor.resolve_team(args.home)
        away = predictor.resolve_team(args.away)
        prediction = predictor.predict(home, away, args.date)
    except (AlreadyPlayedError, UnknownLeagueError, UnknownTeamError) as error:
        raise SystemExit(f"error: {error}") from error
    except PredictionError as error:
        raise SystemExit(f"error: {error}") from error
    except KeyboardInterrupt:
        raise SystemExit("Interrupted.") from None

    print(
        format_prediction(
            prediction,
            explain=args.explain,
            models=selected_models(args.model),
        )
    )


if __name__ == "__main__":
    main()
