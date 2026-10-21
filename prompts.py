"""Prompts sent to the model, and reading the template out of its reply."""
import json
import re

FIRST_PROMPT = """You write Jinja2 templates that turn a data model into network device CLI commands.

Task id: {id}
Task: {description}

The template is rendered with these variables (example values):
{data}

Rules:
- Use the variables. Do not hard-code the example values.
- Output only the commands, one per line, with no comments or explanations.
- Return the whole template inside one ```jinja code block.
"""

RETRY_PROMPT = FIRST_PROMPT + """
Your previous template was:
```jinja
{template}
```

It failed the check "{stage}":
{feedback}

Fix the template and return the whole corrected template inside one ```jinja code block.
"""


def first_prompt(task):
    return FIRST_PROMPT.format(id=task["id"], description=task["description"],
                               data=json.dumps(task["data"], indent=2))


def retry_prompt(task, template, verdict):
    return RETRY_PROMPT.format(id=task["id"], description=task["description"],
                               data=json.dumps(task["data"], indent=2), template=template,
                               stage=verdict.failed_stage, feedback=verdict.feedback)


CODE_BLOCK = re.compile(r"```[a-zA-Z0-9_-]*\n(.*?)```", re.DOTALL)


def extract_template(reply):
    """Take the first code block in the reply, or the whole reply if there is none."""
    match = CODE_BLOCK.search(reply)
    return match.group(1) if match else reply.strip()
