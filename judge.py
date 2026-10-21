"""A staged judge for generated Jinja2 configuration templates.

A candidate template must pass four checks, in order:

1. format      the template is valid Jinja2
2. render      it renders with the example data, without errors
3. correct     the rendered commands match the expected commands
4. generalize  it also gives the right commands for hidden data with the same
               structure, which catches templates that hard-code the example values

The judge stops at the first failed check and explains what went wrong. The
explanation is sent back to the model on a retry, so it never contains the
expected commands.
"""
from dataclasses import dataclass
import difflib

from jinja2 import StrictUndefined, TemplateSyntaxError
from jinja2.sandbox import SandboxedEnvironment

STAGES = ["format", "render", "correct", "generalize"]


@dataclass
class Verdict:
    passed: bool
    stages_passed: int        # 0 to 4
    failed_stage: str | None  # None when passed
    feedback: str             # safe to send back to the model
    similarity: float         # 0.0 to 1.0, line similarity on the example data
    output: str = ""          # commands rendered from the example data


def make_env():
    # Sandboxed: the template comes from a model and is not trusted.
    # StrictUndefined: a misspelled variable is an error, not an empty string.
    return SandboxedEnvironment(undefined=StrictUndefined)


def normalize(text):
    """Commands as a list of lines. Indentation, extra spaces and blank lines are
    ignored, as most network CLIs ignore them too."""
    return [" ".join(line.split()) for line in text.splitlines() if line.strip()]


def mismatch_feedback(expected, actual, label):
    """Describe what is wrong without revealing the expected commands."""
    wrong = [line for line in actual if line not in expected]
    missing = sum(1 for line in expected if line not in actual)
    parts = [f"The commands produced from {label} are not correct."]
    if wrong:
        shown = "\n".join(f"  {line}" for line in wrong[:10])
        parts.append(f"These generated lines are wrong or unexpected:\n{shown}")
    if missing:
        parts.append(f"{missing} expected line(s) are missing.")
    if not wrong and not missing:
        parts.append("All lines exist, but the order or the number of lines is wrong.")
    return "\n".join(parts)


def judge(template, task):
    env = make_env()

    # 1. format
    try:
        env.parse(template)
    except TemplateSyntaxError as e:
        return Verdict(False, 0, "format", f"Template syntax error on line {e.lineno}: {e.message}", 0.0)

    # 2. render
    try:
        output = env.from_string(template).render(**task["data"])
    except Exception as e:  # undefined variable, wrong type, sandbox violation ...
        return Verdict(False, 1, "render", f"Rendering failed: {type(e).__name__}: {e}", 0.0)

    # 3. correct
    expected = normalize("\n".join(task["expected"]))
    actual = normalize(output)
    similarity = difflib.SequenceMatcher(None, expected, actual).ratio()
    if actual != expected:
        feedback = mismatch_feedback(expected, actual, "the example data")
        return Verdict(False, 2, "correct", feedback, similarity, output)

    # 4. generalize
    try:
        hidden_output = env.from_string(template).render(**task["hidden_data"])
    except Exception as e:
        return Verdict(False, 3, "generalize",
                       f"The template failed on other data with the same structure: {type(e).__name__}: {e}",
                       similarity, output)
    hidden_expected = normalize("\n".join(task["hidden_expected"]))
    if normalize(hidden_output) != hidden_expected:
        feedback = ("The output was correct for the example data but wrong for other data with the same "
                    "structure. Use the variables instead of hard-coding the example values.")
        return Verdict(False, 3, "generalize", feedback, similarity, output)

    return Verdict(True, 4, None, "All checks passed.", 1.0, output)
