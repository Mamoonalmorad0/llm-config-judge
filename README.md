# LLM Config Judge

Can a language model write correct network configuration templates, and how do you know when it has?

In this project a model (Google Gemini) writes [Jinja2](https://jinja.palletsprojects.com/) templates that turn a data model, like a list of interfaces or VLANs, into router and switch CLI commands. Every candidate template is checked by a staged judge. If it fails, the judge's feedback goes back to the model, and it gets another try.

## How the judge works

| Stage | Check | Catches |
|---|---|---|
| 1. format | The template is valid Jinja2 | Syntax errors, unclosed blocks |
| 2. render | It renders with the example data in a sandbox | Misspelled or missing variables, unsafe code |
| 3. correct | The rendered commands match the expected commands | Wrong command syntax, wrong order, missing lines |
| 4. generalize | It is also correct for hidden data with the same structure | Templates that hard-code the example values |

The judge stops at the first failed stage. Two design rules keep the evaluation honest:

- The feedback sent to the model never contains the expected commands, only what was wrong in its own output.
- The hidden data in stage 4 is never shown to the model.

The judge is a test oracle, so it has tests too (`tests/test_judge.py`). For example, every expert-written reference template in `tasks/reference/` must pass all four stages.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest                             # test the judge
python run.py --llm fake           # demo with a fake model, no API key needed
```

With a real model (free API key from [Google AI Studio](https://aistudio.google.com/apikey)):

```bash
export GEMINI_API_KEY="your-key"   # never commit the key
python run.py --llm gemini --attempts 3
```

Options: `--model` picks the Gemini model, `--attempts` sets the max attempts per task, `--delay` sets the seconds between calls (default 5), and `--timeout` sets how long to wait for one answer (default 90 s). Each run saves every attempt, template and verdict to `results/`, named after the model.

The free tier allows only about 20 requests per model per day, and one run uses 5 to 15 requests. If the quota runs out, the run stops with a clear message instead of waiting.

## Example output (fake model)

```
Model: fake

task          first attempt         final           attempts
hostname      all checks            all checks             1
interfaces    failed format         all checks             2
vlans         failed generalize     all checks             2
services      failed render         all checks             2
access_list   failed correct        all checks             2

Passed on the first attempt: 1/5
Passed within 3 attempts:  5/5
```

## Results with Gemini (October 2026)

| Run | Model | Judge | Passed on first attempt | Passed within 3 attempts |
|---|---|---|---|---|
| 1 | gemini-3.5-flash | before the fix | 3/5 | 5/5 |
| 2 | gemini-3.5-flash | after the fix | 5/5 | 5/5 |
| 3 | gemini-3.5-flash | after the fix | 5/5 | 5/5 |
| 4 | gemini-3.8-flash | after the fix | 5/5 | 5/5 |

**What I learned**

- In the first run, the model seemed to fail two of five tasks. When I looked at the failures, both were caused by my judge, not the model. The judge rendered templates with Jinja's `trim_blocks` setting, which removes the line break after a block tag such as `{% endif %}`. The model wrote its templates for Jinja's default settings, so two command lines were merged into one.
- After switching the judge to Jinja's default settings, the model passed every task on the first attempt in three runs.
- An automatic judge can report failures that are not the model's fault, so it is worth inspecting the failures before trusting the numbers.
- The current five tasks are too easy to tell models apart. The next step is harder tasks.

## Ideas for next steps

- Harder tasks: nested structures, defaults for missing values, ordering rules, and another vendor's syntax.
- Remove one part of the prompt at a time (task text, example values) and measure the effect.
- Compare models, and compare first-attempt results with results after retries.
- Ask the model to explain and repair broken templates with known faults.
