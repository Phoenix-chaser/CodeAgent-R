from types import SimpleNamespace
import sys
import traceback

import pytest

from agentlab.contracts import Usage
from agentlab.providers.base import APIError, APIRequest, CostEstimate, ModelPricing
from agentlab.providers.openai_compatible import OpenAICompatibleProvider


FAKE_KEY = "offline-key-never-send"


@pytest.fixture(autouse=True)
def offline_provider_environment(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_KEY)
    monkeypatch.setitem(sys.modules, "openai", None)


def make_response():
    return SimpleNamespace(
        id="offline-response",
        model="secondary-model",
        usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=200),
        choices=[SimpleNamespace(
            finish_reason="tool_calls",
            message=SimpleNamespace(
                content=None,
                tool_calls=[SimpleNamespace(
                    id="call-1",
                    function=SimpleNamespace(
                        name="read", arguments='{"rel_path":"src/main.py"}'))]))])


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


def fake_factory(client, constructor_calls):
    def create_client(**kwargs):
        constructor_calls.append(kwargs)
        return client
    return create_client


def pricing(model="secondary-model"):
    return ModelPricing(model, "offline-test-table-v1", 2.0, 4.0)


def test_request_tool_parameters_and_response_are_preserved():
    client = FakeClient()
    constructors = []
    provider = OpenAICompatibleProvider(
        "deepseek", pricing=pricing(), timeout=12.5,
        client_factory=fake_factory(client, constructors))
    messages = [{"role": "user", "content": "read source"}]
    tools = [{"type": "function", "function": {
        "name": "read", "parameters": {"type": "object"}}}]
    request = APIRequest("secondary-model", messages, tools,
                         max_tokens=512, temperature=0.3)
    result = provider.call(request)
    assert constructors == [dict(
        api_key=FAKE_KEY, base_url="https://secondary-api.example/v1",
        timeout=12.5, max_retries=0)]
    assert client.requests == [dict(
        model="secondary-model", messages=messages, tools=tools,
        max_tokens=512, temperature=0.3)]
    assert result.content == ""
    assert result.tool_calls == [{
        "id": "call-1", "name": "read",
        "arguments": '{"rel_path":"src/main.py"}'}]
    assert result.stop_reason == "tool_calls"
    assert result.usage == Usage(input_tokens=1000, output_tokens=200)
    assert result.cost.usd == pytest.approx(0.0028)
    assert result.cost.source == "offline-test-table-v1"
    assert result.raw_response_id == "offline-response"
    assert result.latency_ms >= 0
    assert client.closed


@pytest.mark.parametrize("usage", [
    None,
    Usage(input_tokens=1000),
    Usage(output_tokens=200),
    Usage(),
    Usage(input_tokens=-1, output_tokens=200),
])
def test_incomplete_or_invalid_usage_is_unknown(usage):
    provider = OpenAICompatibleProvider("deepseek", pricing=pricing())
    estimate = provider.estimate_cost(usage)
    assert estimate.usd is None
    assert estimate.source == "unknown"


@pytest.mark.parametrize("raw_usage", [
    None,
    SimpleNamespace(prompt_tokens=1000),
    SimpleNamespace(completion_tokens=200),
])
def test_response_with_missing_usage_fields_remains_unknown(raw_usage):
    response = make_response()
    response.usage = raw_usage
    client = FakeClient(response)
    provider = OpenAICompatibleProvider(
        "deepseek", pricing=pricing(), client_factory=fake_factory(client, []))
    result = provider.call(APIRequest("secondary-model", []))
    assert result.cost.usd is None
    assert result.cost.source == "unknown"
    assert len(client.requests) == 1


@pytest.mark.parametrize("provider_name", ["deepseek",  "primary",  "tertiary"])
def test_no_provider_has_an_implicit_price(provider_name):
    provider = OpenAICompatibleProvider(provider_name)
    estimate = provider.estimate_cost(Usage(input_tokens=1000, output_tokens=200))
    assert estimate.usd is None
    assert estimate.source == "unknown"
    assert CostEstimate().source == "unknown"


def test_pricing_is_bound_to_the_requested_model():
    response = make_response()
    response.model = "other-model"
    client = FakeClient(response)
    provider = OpenAICompatibleProvider(
        "deepseek", pricing=pricing(), client_factory=fake_factory(client, []))
    result = provider.call(APIRequest("other-model", []))
    assert client.requests[0]["model"] == "other-model"
    assert result.model == "other-model"
    assert result.cost.usd is None
    assert result.cost.source == "unknown"


def test_different_response_model_is_not_priced_using_request_alias():
    response = make_response()
    response.model = "different-revision"
    provider = OpenAICompatibleProvider(
        "deepseek", pricing=pricing(),
        client_factory=fake_factory(FakeClient(response), []))
    result = provider.call(APIRequest("secondary-model", []))
    assert result.cost.usd is None
    assert "response model" in result.cost.note


def test_default_model_and_known_zero_cost():
    response = make_response()
    response.model = None
    client = FakeClient(response)
    provider = OpenAICompatibleProvider(
        "deepseek", pricing=ModelPricing("secondary-model", "free-test-v1", 0, 0),
        client_factory=fake_factory(client, []))
    result = provider.call(APIRequest("", []))
    assert client.requests[0]["model"] == "secondary-model"
    assert result.model == "secondary-model"
    assert result.cost.usd == 0
    assert result.cost.source == "free-test-v1"


@pytest.mark.parametrize("field,value", [
    ("model", ""),
    ("source", ""),
    ("source", "unknown"),
    ("input_price_per_m", -1),
    ("input_price_per_m", float("nan")),
    ("output_price_per_m", float("inf")),
    ("output_price_per_m", None),
    ("output_price_per_m", True),
])
def test_pricing_rejects_invalid_model_source_or_rates(field, value):
    arguments = dict(model="secondary-model", source="test-v1",
                     input_price_per_m=2, output_price_per_m=4)
    arguments[field] = value
    with pytest.raises(ValueError):
        ModelPricing(**arguments)


class StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"unsafe response including {FAKE_KEY}")
        self.status_code = status_code


class APIConnectionError(Exception):
    pass


@pytest.mark.parametrize("error,category,retryable,status", [
    (StatusError(429), "rate_limit", True, 429),
    (StatusError(401), "auth", False, 401),
    (StatusError(403), "auth", False, 403),
    (StatusError(408), "network", True, 408),
    (StatusError(503), "server", True, 503),
    (StatusError(400), "unknown", False, 400),
    (ConnectionError(FAKE_KEY), "network", True, None),
    (TimeoutError(FAKE_KEY), "network", True, None),
    (APIConnectionError(FAKE_KEY), "network", True, None),
    (RuntimeError("429 auth rate " + FAKE_KEY), "unknown", False, None),
])
def test_errors_use_status_and_type_without_retry_or_secret_leak(
        error, category, retryable, status):
    client = FakeClient(error=error)
    provider = OpenAICompatibleProvider(
        "deepseek", client_factory=fake_factory(client, []))
    with pytest.raises(APIError) as caught:
        provider.call(APIRequest("secondary-model", []))
    assert caught.value.error_type == category
    assert caught.value.retryable is retryable
    assert caught.value.status_code == status
    assert len(client.requests) == 1
    assert client.closed
    assert FAKE_KEY not in str(caught.value)
    assert FAKE_KEY not in "".join(traceback.format_exception(caught.value))


def test_error_status_can_come_from_response():
    error = RuntimeError(FAKE_KEY)
    error.response = SimpleNamespace(status_code=429)
    provider = OpenAICompatibleProvider(
        "deepseek", client_factory=fake_factory(FakeClient(error=error), []))
    with pytest.raises(APIError) as caught:
        provider.call(APIRequest("secondary-model", []))
    assert caught.value.error_type == "rate_limit"
    assert caught.value.status_code == 429


def test_missing_key_prevents_client_creation(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    constructors = []
    client = FakeClient()
    provider = OpenAICompatibleProvider(
        "deepseek", client_factory=fake_factory(client, constructors))
    with pytest.raises(APIError) as caught:
        provider.call(APIRequest("secondary-model", []))
    assert caught.value.error_type == "auth"
    assert constructors == []
    assert client.requests == []


def test_missing_sdk_reports_dependency_without_network():
    provider = OpenAICompatibleProvider("deepseek")
    with pytest.raises(APIError, match="SDK is not installed"):
        provider.call(APIRequest("secondary-model", []))


def test_empty_choices_are_a_structured_response_error():
    response = make_response()
    response.choices = []
    client = FakeClient(response)
    provider = OpenAICompatibleProvider(
        "deepseek", client_factory=fake_factory(client, []))
    with pytest.raises(APIError, match="no completion choices") as caught:
        provider.call(APIRequest("secondary-model", []))
    assert caught.value.error_type == "server"
    assert caught.value.retryable is False
    assert len(client.requests) == 1
    assert client.closed


def test_malformed_tool_response_is_a_structured_error():
    response = make_response()
    response.choices[0].message.tool_calls = [SimpleNamespace(id="broken")]
    provider = OpenAICompatibleProvider(
        "deepseek", client_factory=fake_factory(FakeClient(response), []))
    with pytest.raises(APIError, match="invalid response"):
        provider.call(APIRequest("secondary-model", []))


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_timeout_must_be_finite_and_positive(timeout):
    with pytest.raises(ValueError, match="timeout"):
        OpenAICompatibleProvider("deepseek", timeout=timeout)
