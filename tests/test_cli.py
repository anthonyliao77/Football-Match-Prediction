"""
Tests for the prediction command line and the interactive session.

The interactive session is the part of this feature a person actually touches,
and it is driven by answering prompts rather than by calling functions. These
tests therefore check the things a person can observe: which answers are
accepted, which are asked again, and that stopping works from every prompt.
The stopping cases are the ones worth having. A prompt that asks again
forever cannot be reported by a test that runs out of scripted answers, so the
answering helper treats an empty script as a failure rather than hanging.
"""

import pytest
from factories import unplayed_pair

import predict
from src.predict import UnknownLeagueError


def an_unplayed(trained_predictor):
    """
    Returns a fixture the data has no result for.

    A season in the test data is partly played, so a fixture named in these
    tests has to be one that is genuinely outstanding. Naming a pairing that
    has been played is not a failure of the session; it is answered, refused
    and offered the next fixture, which is a different test.
    """
    return unplayed_pair(trained_predictor.dataframe.to_dict("records"))


def script(monkeypatch, answers):
    """
    Replies to input() from a fixed list, and fails if the list runs out.

    Running dry is a real outcome to report, not an accident to paper over. A
    session that asks again forever would otherwise spin until the test was
    killed, and the bug would surface as a timeout with nothing to act on.

    Parameters:
        monkeypatch (pytest.MonkeyPatch): The test's patcher.
        answers (list): The answers to give, in order.

    Returns:
        list: The answers as they are given, so a test can check the order.
    """
    remaining = list(answers)
    given = []

    def fake_input(prompt=""):
        """
        Returns the next scripted answer, echoing the prompt as input() does.
        """
        print(prompt, end="", flush=True)

        if not remaining:
            raise AssertionError(
                f"asked for more input than was scripted, after {given}, "
                f"for the prompt {prompt!r}"
            )

        answer = remaining.pop(0)

        given.append(answer)

        return answer

    monkeypatch.setattr("builtins.input", fake_input)

    return given


def test_ask_until_asks_again_until_the_answer_is_accepted(monkeypatch):
    """A rejected answer is asked again rather than ending anything."""
    given = script(monkeypatch, ["no", "nope", "yes"])

    answer = predict.ask_until(
        lambda reply: reply == "yes",
        "Ready? ",
        "  Not that one.",
    )

    assert answer == "yes"
    assert given == ["no", "nope", "yes"]


def test_ask_until_hands_back_an_exit_word_straight_away(monkeypatch):
    """An exit word returns without the predicate ever being consulted.

    The predicate rejects everything here. Were the exit word tested only
    against the predicate it would be indistinguishable from a wrong answer,
    and ask_until would ask again instead of letting the caller stop.
    """
    given = script(monkeypatch, ["quit"])

    assert predict.ask_until(
        lambda reply: False,
        "Anything? ",
        "  Not accepted.",
    ) == "quit"

    assert given == ["quit"]


def test_ask_until_stops_asking_once_an_exit_word_arrives(monkeypatch):
    """A wrong answer followed by an exit word ends the run, not the loop."""
    script(monkeypatch, ["nonsense", "exit"])

    assert predict.ask_until(
        lambda reply: reply == "yes",
        "Ready? ",
        "  Try again.",
    ) == "exit"


@pytest.mark.parametrize(
    "word",
    ["quit", "QUIT", "Quit", "exit", "EXIT", "Exit"],
)
def test_exit_words_are_recognised_whatever_the_case(monkeypatch, word):
    """A person typing QUIT gets the same answer as one typing quit."""
    given = script(monkeypatch, [word])

    assert predict.ask_until(
        lambda reply: False,
        "Anything? ",
        "",
    ) == word

    assert given == [word]


def test_stopping_at_the_league_prompt_ends_the_session(monkeypatch):
    """Stopping before naming a league never loads any data."""
    script(monkeypatch, ["quit"])

    predict.run_interactive()


def test_stopping_at_the_home_prompt_ends_the_session(monkeypatch, trained):
    """Stopping after the league has been built still ends the session."""
    script(monkeypatch, ["PremierLeague", "quit"])

    predict.run_interactive()


def test_stopping_at_the_away_prompt_ends_the_session(monkeypatch, trained):
    """Stopping once both teams are named still ends the session."""
    script(monkeypatch, ["PremierLeague", "Arsenal", "quit"])

    predict.run_interactive()


def test_stopping_at_the_date_prompt_ends_the_session(monkeypatch, trained):
    """A fully named fixture can still be abandoned at the date."""
    script(monkeypatch, ["PremierLeague", "Arsenal", "Chelsea", "quit"])

    predict.run_interactive()


def test_stopping_instead_of_answering_the_continue_question_ends_it(
    monkeypatch, trained
):
    """An exit word in place of an answer leaves rather than asking again."""
    script(monkeypatch, ["PremierLeague", "Arsenal", "Chelsea", "", "quit"])

    predict.run_interactive()


def test_answering_no_to_the_continue_question_ends_the_session(
    monkeypatch, trained
):
    """A plain no is enough to leave."""
    home, away = an_unplayed(trained)

    script(monkeypatch, ["PremierLeague", home, away, "", "no"])

    predict.run_interactive()


def test_a_league_that_is_not_covered_is_asked_again(monkeypatch):
    """An unknown league is refused, and the prompt comes round again."""
    given = script(monkeypatch, ["Conference", "quit"])

    predict.run_interactive()

    assert given == ["Conference", "quit"]


def test_a_league_can_be_typed_loosely(monkeypatch, trained):
    """A league is matched ignoring case, spacing and the name's aliases."""
    home, away = an_unplayed(trained)

    script(monkeypatch, ["epl", home, away, "", "n"])

    predict.run_interactive()


def test_a_team_can_be_typed_loosely(monkeypatch, trained, capsys):
    """A team's name is matched ignoring case and surrounding space."""
    home, away = an_unplayed(trained)

    script(monkeypatch, [
        "PremierLeague", f"  {home.lower()} ", away, "", "n",
    ])

    predict.run_interactive()

    assert f"{home} v {away}" in capsys.readouterr().out


def test_the_end_of_the_input_ends_the_session(monkeypatch, trained, capsys):
    """An input that ends stops the session instead of raising.

    Piped input running out, or Ctrl-D at a terminal, is a way of stopping. A
    traceback for it would be the least useful thing the tool could do.
    """
    def no_more_input(prompt=""):
        """
        Reports the end of the input.
        """
        raise EOFError

    monkeypatch.setattr("builtins.input", no_more_input)

    predict.run_interactive()

    assert "Traceback" not in capsys.readouterr().out


def test_an_interrupted_prompt_ends_the_session(monkeypatch, trained, capsys):
    """Ctrl-C at a prompt stops the session instead of raising."""
    def interrupted(prompt=""):
        """
        Reports an interrupt.
        """
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupted)

    predict.run_interactive()

    assert "Traceback" not in capsys.readouterr().out


def test_a_command_line_prediction_can_be_interrupted(monkeypatch):
    """Ctrl-C while a command line prediction is loading exits cleanly."""
    def interrupted(league):
        """
        Reports an interrupt raised while the league is being loaded.
        """
        raise KeyboardInterrupt

    monkeypatch.setattr(predict, "Predictor", interrupted)

    with pytest.raises(SystemExit) as stopped:
        predict.main([
            "--league", "PremierLeague", "--home", "Arsenal",
            "--away", "Chelsea",
        ])

    assert "Interrupted" in str(stopped.value)


def test_a_named_fixture_is_predicted_and_offered_another(
    monkeypatch, trained, capsys
):
    """The ordinary path: name a fixture, get a prediction, get asked again."""
    home, away = an_unplayed(trained)

    given = script(monkeypatch, ["PremierLeague", home, away, "", "n"])

    predict.run_interactive()

    output = capsys.readouterr().out

    assert f"{home} v {away}" in output
    assert "Another fixture?" in output
    assert given == ["PremierLeague", home, away, "", "n"]


def test_a_fixture_that_was_already_played_is_reported_not_fatal(
    monkeypatch, trained, capsys
):
    """Asking about a decided fixture explains itself and keeps going.

    A season has already been played, and the model has no opinion about it, so
    the session says so and comes back to ask for another fixture rather than
    raising.
    """
    played = trained.dataframe.iloc[0]
    home, away = an_unplayed(trained)

    script(monkeypatch, [
        "PremierLeague",
        played["HomeTeam"],
        played["AwayTeam"],
        played["Date"].strftime("%Y-%m-%d"),
        "PremierLeague",
        home,
        away,
        "",
        "n",
    ])

    predict.run_interactive()

    output = capsys.readouterr().out

    assert "has already been played" in output
    assert f"{home} v {away}" in output, "the session should carry on"


def test_an_unreadable_date_is_reported_and_offered_again(
    monkeypatch, trained, capsys
):
    """A date that is not a date is refused, and explained rather than raised.

    The fixture is not lost to the bad date: the session reports it, comes back
    to the league prompt, and answers the same fixture once the date is blank.
    """
    home, away = an_unplayed(trained)

    script(monkeypatch, [
        "PremierLeague", home, away, "next Tuesday",
        "PremierLeague", home, away, "",
        "n",
    ])

    predict.run_interactive()

    output = capsys.readouterr().out

    assert "Could not read" in output
    assert output.count(f"{home} v {away}") == 1, "only the good date predicts"


def test_a_fixture_can_be_predicted_twice_in_one_session(
    monkeypatch, trained, capsys
):
    """The league's models are kept, so a second fixture is cheap."""
    first = an_unplayed(trained)
    second = an_unplayed(trained)

    script(monkeypatch, [
        "PremierLeague", *first, "", "y",
        "PremierLeague", *second, "", "n",
    ])

    predict.run_interactive()

    output = capsys.readouterr().out

    assert output.count("Random Forest") == 2
    assert output.count("Loading PremierLeague and training") == 1


def test_a_fixture_can_be_predicted_with_a_date_given(monkeypatch, trained):
    """An explicit date is used instead of the default."""
    home, away = an_unplayed(trained)

    script(monkeypatch, ["PremierLeague", home, away, "2025-03-01", "n"])

    predict.run_interactive()


def test_a_half_named_fixture_on_the_command_line_is_refused():
    """A fixture cannot be half-named on the command line."""
    with pytest.raises(SystemExit) as stopped:
        predict.main(["--home", "Arsenal"])

    assert "--away" in str(stopped.value)


def test_a_command_line_without_a_league_is_refused():
    """A command line with no league is an error rather than a guess.

    Defaulting the league would answer a different question from the one asked,
    and the answer would look entirely ordinary.
    """
    with pytest.raises(SystemExit) as stopped:
        predict.main(["--home", "Arsenal", "--away", "Chelsea"])

    assert "--league" in str(stopped.value)


def test_a_command_line_with_no_arguments_starts_a_session(monkeypatch):
    """Running it with no arguments is the way into the session."""
    script(monkeypatch, ["quit"])

    predict.main([])


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("both", ["Random Forest", "XGBoost"]),
        ("rf", ["Random Forest"]),
        ("xgb", ["XGBoost"]),
    ],
)
def test_the_model_flag_chooses_which_model_is_shown(choice, expected):
    """The model flag maps onto the labels that get displayed."""
    assert predict.selected_models(choice) == expected


def test_a_league_name_that_is_not_covered_fails_with_the_alternatives():
    """An unusable league name fails, and lists what is available."""
    with pytest.raises(UnknownLeagueError) as error:
        predict.resolve_league("Conference")

    assert "PremierLeague" in str(error.value)
