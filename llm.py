"""Model clients. Each client has one method: ask(prompt) -> reply text."""
from collections import defaultdict
import os
import re


class GeminiLLM:
    """Google Gemini. Reads the API key from the GEMINI_API_KEY environment variable."""

    def __init__(self, model=None):
        from google import genai  # pip install google-genai

        self.client = genai.Client()
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")

    def ask(self, prompt):
        if hasattr(self.client, "interactions"):  # current SDK
            reply = self.client.interactions.create(model=self.model, input=prompt)
            return reply.output_text
        reply = self.client.models.generate_content(model=self.model, contents=prompt)  # older SDK
        return reply.text


class FakeLLM:
    """Returns prepared replies, so the pipeline can be tested without an API key.

    replies maps a task id to a list of replies, one per attempt. The last reply
    is repeated if there are more attempts than replies."""

    def __init__(self, replies):
        self.replies = replies
        self.calls = defaultdict(int)

    def ask(self, prompt):
        task_id = re.search(r"Task id: (\S+)", prompt).group(1)
        answers = self.replies[task_id]
        reply = answers[min(self.calls[task_id], len(answers) - 1)]
        self.calls[task_id] += 1
        return reply
