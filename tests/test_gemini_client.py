"""Tests for the Gemini client against a small local fake server (no real API calls)."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading

from llm import GeminiLLM, LLMError


def fake_server(status, payload):
    """Start a local server that answers every POST with `status` and `payload`."""
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"] = self.path
            seen["key"] = self.headers.get("x-goog-api-key")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


def client(server, monkeypatch_env):
    monkeypatch_env("GEMINI_API_KEY", "test-key")
    monkeypatch_env("GEMINI_API_BASE", f"http://127.0.0.1:{server.server_port}")
    return GeminiLLM("test-model", timeout=5)


def env_setter():
    import os
    old = {}

    def setenv(name, value):
        old.setdefault(name, os.environ.get(name))
        os.environ[name] = value

    def restore():
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    return setenv, restore


def test_reads_the_reply_text_and_skips_thoughts():
    server, seen = fake_server(200, {"candidates": [{"content": {"parts": [
        {"text": "thinking...", "thought": True}, {"text": "```jinja\nhostname {{ hostname }}\n```"}]}}]})
    setenv, restore = env_setter()
    try:
        reply = client(server, setenv).ask("hello")
    finally:
        restore()
        server.shutdown()
    assert reply == "```jinja\nhostname {{ hostname }}\n```"
    assert seen["path"] == "/models/test-model:generateContent"
    assert seen["key"] == "test-key"
    assert seen["body"]["contents"][0]["parts"][0]["text"] == "hello"


def check_error(status, api_status, expected_words):
    server, _ = fake_server(status, {"error": {"code": status, "status": api_status, "message": "Something happened."}})
    setenv, restore = env_setter()
    try:
        client(server, setenv).ask("hello")
        raised = None
    except LLMError as e:
        raised = str(e)
    finally:
        restore()
        server.shutdown()
    assert raised is not None
    for word in expected_words:
        assert word in raised, raised


def test_quota_error_is_reported_at_once():
    check_error(429, "RESOURCE_EXHAUSTED", ["HTTP 429", "RESOURCE_EXHAUSTED", "quota"])


def test_overloaded_server_is_reported_at_once():
    check_error(503, "UNAVAILABLE", ["HTTP 503", "UNAVAILABLE", "overloaded"])


def test_missing_api_key_gives_a_clear_message():
    setenv, restore = env_setter()
    import os
    old = os.environ.pop("GEMINI_API_KEY", None)
    try:
        GeminiLLM("test-model")
        raised = None
    except LLMError as e:
        raised = str(e)
    finally:
        if old is not None:
            os.environ["GEMINI_API_KEY"] = old
        restore()
    assert raised and "GEMINI_API_KEY" in raised
