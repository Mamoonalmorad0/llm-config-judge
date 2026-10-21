import json
from pathlib import Path

import pytest

from judge import judge, normalize
from llm import FakeLLM
from prompts import extract_template, first_prompt
from run import solve

ROOT = Path(__file__).parent.parent
TASKS = json.loads((ROOT / "tasks" / "tasks.json").read_text(encoding="utf-8"))
BY_ID = {t["id"]: t for t in TASKS}


def reference(task_id):
    return (ROOT / "tasks" / "reference" / f"{task_id}.j2").read_text(encoding="utf-8")


# The judge is a test oracle, so it needs tests of its own.
# First: every expert-written reference template must pass every check.
@pytest.mark.parametrize("task", TASKS, ids=[t["id"] for t in TASKS])
def test_reference_templates_pass_all_checks(task):
    verdict = judge(reference(task["id"]), task)
    assert verdict.passed, verdict.feedback
    assert verdict.stages_passed == 4


def test_syntax_error_fails_format():
    broken = reference("interfaces").replace("{% endif %}", "")
    verdict = judge(broken, BY_ID["interfaces"])
    assert verdict.failed_stage == "format"
    assert verdict.stages_passed == 0


def test_misspelled_variable_fails_render():
    verdict = judge("hostname {{ host_name }}", BY_ID["hostname"])
    assert verdict.failed_stage == "render"


def test_wrong_command_fails_correct_without_leaking_the_answer():
    wrong = reference("hostname").replace("ip domain-name", "ip domain name")
    verdict = judge(wrong, BY_ID["hostname"])
    assert verdict.failed_stage == "correct"
    assert "ip domain name lab.local" in verdict.feedback   # the wrong line is shown
    assert "ip domain-name lab.local" not in verdict.feedback  # the right line is not
    assert 0 < verdict.similarity < 1


def test_hard_coded_values_fail_generalize():
    hard_coded = "hostname edge-router-1\nip domain-name lab.local\n"
    verdict = judge(hard_coded, BY_ID["hostname"])
    assert verdict.failed_stage == "generalize"
    assert verdict.stages_passed == 3


def test_sandbox_blocks_unsafe_templates():
    unsafe = "{{ ''.__class__.__mro__[1].__subclasses__() }}"
    verdict = judge(unsafe, BY_ID["hostname"])
    assert not verdict.passed
    assert verdict.failed_stage == "render"


def test_normalize_ignores_indentation_extra_spaces_and_blank_lines():
    assert normalize("interface Gi0/1\n\n    description   Office LAN \n") == [
        "interface Gi0/1", "description Office LAN"]


def test_extract_template_from_code_block_or_plain_text():
    assert extract_template("Here you go:\n```jinja\nhostname {{ hostname }}\n```\nDone.") == "hostname {{ hostname }}\n"
    assert extract_template("hostname {{ hostname }}") == "hostname {{ hostname }}"


def test_prompt_does_not_contain_expected_commands():
    prompt = first_prompt(BY_ID["interfaces"])
    assert "ip address 203.0.113.2 255.255.255.252" not in prompt  # no expected command
    assert "TenGigabitEthernet1/0/1" not in prompt  # hidden data stays hidden


def test_solve_retries_with_feedback_until_it_passes():
    task = BY_ID["vlans"]
    hard_coded = "```jinja\nvlan 10\n name users\nvlan 20\n name voice\nvlan 99\n name management\n```"
    llm = FakeLLM({"vlans": [hard_coded, f"```jinja\n{reference('vlans')}```"]})
    attempts = solve(llm, task, max_attempts=3)
    assert [a["passed"] for a in attempts] == [False, True]
    assert attempts[0]["failed_stage"] == "generalize"


def test_solve_stops_after_max_attempts():
    llm = FakeLLM({"hostname": ["```jinja\nhostname {{ host }}\n```"]})
    attempts = solve(llm, BY_ID["hostname"], max_attempts=2)
    assert len(attempts) == 2
    assert not attempts[-1]["passed"]
