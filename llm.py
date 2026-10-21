"""Model clients. Each client has one method: ask(prompt) -> reply text."""
from collections import defaultdict
import json
import os
import re
import urllib.error
import urllib.request

GEMINI_API = "https://generativelanguage.googleapis.com/v1beta"

HINTS = {
    429: "the free quota for this model is used up; try another model or wait until it resets",
    503: "Google's servers for this model are overloaded; try again later or use another model",
    404: "the model name is wrong or the model is no longer available; check --model",
    400: "the request was rejected; check that GEMINI_API_KEY is a valid key",
    403: "the API key is not allowed to use this model",
}


class LLMError(RuntimeError):
    """An error from the model API, with a short explanation of what to do."""


class GeminiLLM:
    """Google Gemini over its REST API. Reads the key from GEMINI_API_KEY.

    The request is made directly, without retries, so a quota or server error
    is reported at once instead of being retried quietly in the background."""

    def __init__(self, model=None, timeout=90.0):
        self.key = os.environ.get("GEMINI_API_KEY")
        if not self.key:
            raise LLMError("GEMINI_API_KEY is not set. Run: export GEMINI_API_KEY=\"your-key\"")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
        self.base = os.environ.get("GEMINI_API_BASE", GEMINI_API)  # overridable for tests
        self.timeout = timeout

    def ask(self, prompt):
        url = f"{self.base}/models/{self.model}:generateContent"
        body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
        request = urllib.request.Request(url, data=body, method="POST", headers={
            "Content-Type": "application/json",
            "x-goog-api-key": self.key,
        })
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.load(response)
        except urllib.error.HTTPError as e:
            try:
                error = json.load(e).get("error", {})
            except ValueError:
                error = {}
            status = error.get("status", "")
            message = (error.get("message") or e.reason or "").split("\n")[0]
            hint = HINTS.get(e.code, "")
            raise LLMError(f"HTTP {e.code} {status}: {message}" + (f" ({hint})" if hint else "")) from None
        except urllib.error.URLError as e:
            raise LLMError(f"Could not reach the API: {e.reason} (check the network)") from None

        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError):
            reason = (data.get("candidates") or [{}])[0].get("finishReason", "unknown")
            raise LLMError(f"The model returned no text (finish reason: {reason})") from None
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))


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
