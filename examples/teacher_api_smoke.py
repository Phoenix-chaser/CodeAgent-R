"""Send one 教师模型 connectivity request using a standard API account."""
from __future__ import annotations

import argparse
import json
import os


PLATFORMS = {
    "primary": ("https://teacher-api.example/api/paas/v4", "TEACHER_API_KEY"),
    "secondary": ("https://teacher-api.example/secondary/v4", "TEACHER_API_KEY_2"),
}
MODEL = "teacher-api-model"


def positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def token_count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def usage_summary(usage) -> dict | None:
    if usage is None:
        return None
    completion_details = getattr(usage, "completion_tokens_details", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    return {
        "input_tokens": token_count(getattr(usage, "prompt_tokens", None)),
        "output_tokens": token_count(getattr(usage, "completion_tokens", None)),
        "total_tokens": token_count(getattr(usage, "total_tokens", None)),
        "reasoning_tokens": token_count(getattr(completion_details, "reasoning_tokens", None)),
        "cached_input_tokens": token_count(getattr(prompt_details, "cached_tokens", None)),
    }


def emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None, *, client_factory=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=sorted(PLATFORMS))
    parser.add_argument("--max-tokens", type=positive_integer, default=1024)
    arguments = parser.parse_args(argv)
    base_url, key_name = PLATFORMS[arguments.platform]
    api_key = os.environ.get(key_name, "").strip()
    if not api_key:
        emit({"status": "configuration_error", "error": f"Set {key_name} before calling the API."})
        return 2
    if client_factory is None:
        try:
            from openai import OpenAI
        except ImportError:
            emit({"status": "configuration_error", "error": "Install the project's api extra first."})
            return 2
        client_factory = OpenAI

    client = None
    try:
        client = client_factory(api_key=api_key, base_url=base_url,
                                timeout=60, max_retries=0)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: API_OK"}],
            temperature=1.0,
            max_tokens=arguments.max_tokens,
            extra_body={"thinking": {"type": "enabled"}, "reasoning_effort": "low"},
        )
        choice = response.choices[0]
        content = getattr(choice.message, "content", None)
        content = content if isinstance(content, str) else ""
        stop_reason = getattr(choice, "finish_reason", None)
        complete = stop_reason == "stop" and bool(content.strip())
        payload = {
            "status": "complete" if complete else "incomplete",
            "content": content,
            "model": getattr(response, "model", None),
            "stop_reason": stop_reason,
            "usage": usage_summary(getattr(response, "usage", None)),
            "response_id": getattr(response, "id", None),
            "cost": {"usd": None, "source": "unknown"},
            "training_data_eligible": False,
        }
        if stop_reason == "length":
            payload["note"] = "Output limit reached; this incomplete response is not training data. No retry was sent."
        elif not complete:
            payload["note"] = "No complete text response was returned. No retry was sent."
        emit(payload)
        return 0 if complete else 3
    except Exception as error:
        status_code = getattr(error, "status_code", None)
        emit({
            "status": "request_error",
            "error": "The API request or response failed. Check account access, endpoint, model, quota, and network. No automatic retry was sent.",
            "http_status": status_code if isinstance(status_code, int) and not isinstance(status_code, bool) else None,
        })
        return 1
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
