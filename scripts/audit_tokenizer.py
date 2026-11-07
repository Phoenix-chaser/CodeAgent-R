"""Validate production assistant-only masking with a real model tokenizer."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agentlab.training.hf_sft import SFTDataset, _as_list


def audit_cases():
    return [
        ("chinese_code", [
            {"role": "system", "content": "你是代码修复专家。"},
            {"role": "user", "content": "修复 None 边界。"},
        ], "def parse(value):\n    return None if value is None else int(value)\n"),
        ("multiturn_history", [
            {"role": "system", "content": "Output Python."},
            {"role": "user", "content": "Fix it."},
            {"role": "assistant", "content": "def old():\n    return 1\n"},
            {"role": "user", "content": "错误：期望返回「你好」。"},
        ], 'def fixed():\n    return "你好"\n'),
        ("unicode_indentation", [
            {"role": "user", "content": "Fix Unicode and tabs."},
        ], 'def café():\n\treturn "é🙂"\n'),
    ]


def audit_tokenizer(tokenizer, model: str, revision: str | None) -> dict:
    records = [
        {"sample_id": name, "task_id": name, "source_group": "tokenizer_audit",
         "split": "train", "messages": messages,
         "target": {"role": "assistant", "content": target},
         "provenance": {"strict_pass": True}}
        for name, messages, target in audit_cases()
    ]
    with tempfile.TemporaryDirectory(prefix="codeagent-tokenizer-") as temporary:
        path = Path(temporary) / "template-probes.jsonl"
        path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n"
                                for record in records), encoding="utf-8")
        dataset = SFTDataset(path, tokenizer)
    if len(dataset) != len(records):
        raise AssertionError("tokenizer audit unexpectedly dropped a probe")
    checks = []
    for sample, record in zip(dataset, records):
        prefix = _as_list(tokenizer.apply_chat_template(
            record["messages"], tokenize=True, add_generation_prompt=True,
            return_tensors=None))
        full = _as_list(tokenizer.apply_chat_template(
            record["messages"] + [record["target"]], tokenize=True,
            add_generation_prompt=False, return_tensors=None))
        labels = sample["labels"]
        if sample["input_ids"] != full or labels[:len(prefix)] != [-100] * len(prefix):
            raise AssertionError("context or token stream differs from model template")
        if labels[len(prefix):] != full[len(prefix):]:
            raise AssertionError("target mask does not cover the complete template response")
        supervised_text = tokenizer.decode(full[len(prefix):], skip_special_tokens=False)
        if not supervised_text.startswith(record["target"]["content"]):
            raise AssertionError("decoded target differs from the intended assistant response")
        checks.append({"name": record["sample_id"], "passed": True,
                       "input_tokens": len(full), "masked_context_tokens": len(prefix),
                       "supervised_tokens": len(full) - len(prefix),
                       "supervised_text": supervised_text})
    return {
        "schema_version": 1, "model": model, "requested_revision": revision,
        "transformers_version": importlib.metadata.version("transformers"),
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
        "all_passed": True, "checks": checks,
        "scope": "Tokenizer and mask validation only; no model weights or training examples exported",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--revision", default="c03e6d358207e414f1eca0bb1891e29f1db0e242")
    parser.add_argument("--output", default="results/tokenizer-audit.json")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                             trust_remote_code=False)
    report = audit_tokenizer(tokenizer, args.model, args.revision)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
