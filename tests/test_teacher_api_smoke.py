import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "teacher_api_smoke.py"
SPEC = importlib.util.spec_from_file_location("teacher_api_smoke", SCRIPT)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)
FAKE_KEY = "offline-key-do-not-send"


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    monkeypatch.delenv("TEACHER_API_KEY", raising=False)
    monkeypatch.delenv("TEACHER_API_KEY_2", raising=False)
    monkeypatch.setitem(sys.modules, "openai", None)


def make_response(stop_reason="stop", content="API_OK"):
    return SimpleNamespace(
        id="offline-response",
        model="teacher-api-model",
        choices=[SimpleNamespace(
            finish_reason=stop_reason,
            message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=20, total_tokens=32,
                              completion_tokens_details=SimpleNamespace(reasoning_tokens=15)),
    )


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response if response is not None else make_response()
        self.error = error
        self.requests = []
        self.closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response

    def close(self):
        self.closed = True


def make_factory(client, calls):
    def factory(**kwargs):
        calls.append(kwargs)
        return client
    return factory


@pytest.mark.parametrize("platform,key_name,base_url", [
    ("primary", "TEACHER_API_KEY", "https://teacher-api.example/api/paas/v4"),
    ("secondary", "TEACHER_API_KEY_2", "https://teacher-api.example/secondary/v4"),
])
def test_one_request_uses_platform_and_thinking_configuration(
        platform, key_name, base_url, monkeypatch, capsys):
    monkeypatch.setenv(key_name, FAKE_KEY)
    client = FakeClient()
    calls = []
    result = smoke.main(["--platform", platform], client_factory=make_factory(client, calls))
    assert result == 0
    assert calls == [{"api_key": FAKE_KEY, "base_url": base_url,
                      "timeout": 60, "max_retries": 0}]
    assert client.requests == [{
        "model": "teacher-api-model",
        "messages": [{"role": "user", "content": "Reply with exactly: API_OK"}],
        "temperature": 1.0,
        "max_tokens": 1024,
        "extra_body": {"thinking": {"type": "enabled"}, "reasoning_effort": "low"},
    }]
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert FAKE_KEY not in output
    assert payload["content"] == "API_OK"
    assert payload["model"] == "teacher-api-model"
    assert payload["response_id"] == "offline-response"
    assert payload["stop_reason"] == "stop"
    assert payload["usage"]["reasoning_tokens"] == 15
    assert payload["cost"] == {"usd": None, "source": "unknown"}
    assert payload["training_data_eligible"] is False
    assert client.closed


def test_missing_key_prevents_client_creation(capsys):
    client = FakeClient()
    calls = []
    assert smoke.main(["--platform", "primary"],
                      client_factory=make_factory(client, calls)) == 2
    assert calls == []
    assert client.requests == []
    assert json.loads(capsys.readouterr().out)["status"] == "configuration_error"


@pytest.mark.parametrize("content", ["partial", None])
def test_length_response_is_incomplete_without_retry(content, monkeypatch, capsys):
    monkeypatch.setenv("TEACHER_API_KEY", FAKE_KEY)
    client = FakeClient(make_response("length", content))
    assert smoke.main(["--platform", "primary", "--max-tokens", "2048"],
                      client_factory=make_factory(client, [])) == 3
    assert len(client.requests) == 1
    assert client.requests[0]["max_tokens"] == 2048
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "incomplete"
    assert payload["training_data_eligible"] is False
    assert client.closed


@pytest.mark.parametrize("stop_reason,content", [("stop", ""), ("content_filter", None)])
def test_other_incomplete_response_is_not_success(stop_reason, content, monkeypatch, capsys):
    monkeypatch.setenv("TEACHER_API_KEY_2", FAKE_KEY)
    client = FakeClient(make_response(stop_reason, content))
    assert smoke.main(["--platform", "secondary"], client_factory=make_factory(client, [])) == 3
    assert len(client.requests) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "incomplete"


def test_exception_is_redacted_without_retry(monkeypatch, capsys):
    monkeypatch.setenv("TEACHER_API_KEY", FAKE_KEY)
    client = FakeClient(error=RuntimeError("unsafe secret: " + FAKE_KEY))
    assert smoke.main(["--platform", "primary"],
                      client_factory=make_factory(client, [])) == 1
    output = capsys.readouterr().out
    assert FAKE_KEY not in output
    assert "unsafe secret" not in output
    assert json.loads(output)["status"] == "request_error"
    assert len(client.requests) == 1
    assert client.closed


def test_help_does_not_require_sdk_or_key(capsys):
    with pytest.raises(SystemExit) as caught:
        smoke.main(["--help"])
    assert caught.value.code == 0
    assert "--platform" in capsys.readouterr().out


def test_missing_sdk_is_configuration_error(monkeypatch, capsys):
    monkeypatch.setenv("TEACHER_API_KEY", FAKE_KEY)
    assert smoke.main(["--platform", "primary"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "configuration_error"


@pytest.mark.parametrize("value", ["0", "-1", "1.5"])
def test_max_tokens_must_be_positive_integer(value):
    with pytest.raises(SystemExit) as caught:
        smoke.main(["--platform", "primary", "--max-tokens", value])
    assert caught.value.code == 2


def test_platform_is_required():
    with pytest.raises(SystemExit) as caught:
        smoke.main([])
    assert caught.value.code == 2
