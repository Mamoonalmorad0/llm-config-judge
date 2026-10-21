"""Let a model write a template for each task, judge it, and retry with feedback.

    python run.py --llm fake      demo without an API key
    python run.py --llm gemini    real run (needs GEMINI_API_KEY)
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import threading
import time

from judge import STAGES, judge
from llm import FakeLLM, GeminiLLM, LLMError
from prompts import extract_template, first_prompt, retry_prompt

ROOT = Path(__file__).parent


def load_tasks(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ask_with_timeout(llm, prompt, timeout):
    """Ask the model, but give up after `timeout` seconds. The Gemini library
    retries quietly on errors such as an exhausted quota, which can look like a hang."""
    result = {}

    def call():
        try:
            result["reply"] = llm.ask(prompt)
        except Exception as e:
            result["error"] = e

    worker = threading.Thread(target=call, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(f"no answer within {timeout:.0f} s (quota used up, or no network?)")
    if "error" in result:
        raise result["error"]
    return result["reply"]


def solve(llm, task, max_attempts, delay=0.0, timeout=90.0):
    """Ask, judge, and retry with the judge's feedback until the template passes."""
    attempts = []
    prompt = first_prompt(task)
    for number in range(1, max_attempts + 1):
        try:
            reply = ask_with_timeout(llm, prompt, timeout)
        except Exception as e:  # network error, rate limit, timeout ...
            attempts.append({"attempt": number, "error": f"{type(e).__name__}: {e}"})
            break
        template = extract_template(reply)
        verdict = judge(template, task)
        attempts.append({
            "attempt": number,
            "template": template,
            "passed": verdict.passed,
            "stages_passed": verdict.stages_passed,
            "failed_stage": verdict.failed_stage,
            "feedback": verdict.feedback,
            "similarity": round(verdict.similarity, 2),
        })
        if verdict.passed:
            break
        prompt = retry_prompt(task, template, verdict)
        time.sleep(delay)
    return attempts


def demo_llm():
    """A fake model that makes one typical mistake per task before getting it right."""
    ref = {p.stem: p.read_text(encoding="utf-8") for p in (ROOT / "tasks" / "reference").glob("*.j2")}

    def block(text):
        return f"```jinja\n{text}```"

    return FakeLLM({
        "hostname": [block(ref["hostname"])],
        "interfaces": [block(ref["interfaces"].replace("{% endif %}\n", "")), block(ref["interfaces"])],
        "vlans": [block("vlan 10\n name users\nvlan 20\n name voice\nvlan 99\n name management\n"),
                  block(ref["vlans"])],
        "services": [block(ref["services"].replace("in ntp_servers", "in ntp")), block(ref["services"])],
        "access_list": [block(ref["access_list"].replace('" eq " ~ rule.port', '" port " ~ rule.port')),
                        block(ref["access_list"])],
    })


def stage_name(stages_passed):
    return "all checks" if stages_passed == len(STAGES) else f"failed {STAGES[stages_passed]}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", choices=["fake", "gemini"], default="fake")
    parser.add_argument("--model", help="model name, e.g. gemini-3.5-flash")
    parser.add_argument("--attempts", type=int, default=3, help="max attempts per task")
    parser.add_argument("--delay", type=float, default=None, help="seconds between calls")
    parser.add_argument("--timeout", type=float, default=90.0, help="max seconds to wait for one answer")
    parser.add_argument("--tasks", default=str(ROOT / "tasks" / "tasks.json"))
    args = parser.parse_args()

    try:
        llm = demo_llm() if args.llm == "fake" else GeminiLLM(args.model, timeout=args.timeout)
    except LLMError as e:
        raise SystemExit(f"Error: {e}")
    delay = args.delay if args.delay is not None else (0.0 if args.llm == "fake" else 5.0)
    tasks = load_tasks(args.tasks)

    results = []
    stopped = None
    print(f"Model: {getattr(llm, 'model', 'fake')}\n")
    print(f"{'task':<14}{'first attempt':<22}{'final':<16}{'attempts':>8}")
    for task in tasks:
        print(f"{task['id']:<14}", end="", flush=True)  # shows which task is running
        attempts = solve(llm, task, args.attempts, delay, args.timeout)
        results.append({"task": task["id"], "attempts": attempts})
        if "error" in attempts[-1]:
            print("error")
            stopped = attempts[-1]["error"]
            break
        first = stage_name(attempts[0]["stages_passed"])
        final = stage_name(attempts[-1]["stages_passed"])
        print(f"{first:<22}{final:<16}{len(attempts):>8}")
        time.sleep(delay)

    if stopped:
        print(f"\nStopped, the model did not answer: {stopped}")
        print("Results are saved, but this run is incomplete. Do not count it.")
    else:
        n = len(results)
        first_ok = sum(1 for r in results if r["attempts"][0].get("passed"))
        final_ok = sum(1 for r in results if r["attempts"][-1].get("passed"))
        print(f"\nPassed on the first attempt: {first_ok}/{n}")
        word = "attempt" if args.attempts == 1 else "attempts"
        print(f"Passed within {args.attempts} {word}:  {final_ok}/{n}")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model = getattr(llm, "model", "fake")
    name = f"{stamp}-{model}" + ("-incomplete" if stopped else "")
    base, i = name, 2
    while (out_dir / f"{name}.json").exists():  # two runs in the same second
        name, i = f"{base}-{i}", i + 1
    (out_dir / f"{name}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with open(out_dir / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "attempt", "passed", "stages_passed", "failed_stage", "similarity", "error"])
        for r in results:
            for a in r["attempts"]:
                writer.writerow([r["task"], a["attempt"], a.get("passed"), a.get("stages_passed"),
                                 a.get("failed_stage"), a.get("similarity"), a.get("error")])
    print(f"Saved results/{name}.json and results/{name}.csv")


if __name__ == "__main__":
    main()
