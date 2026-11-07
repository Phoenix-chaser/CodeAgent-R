"""Regression tests for the production SFT data and optimization helpers."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from agentlab.training.hf_sft import (
    Collator, SFTDataset, evaluate_loss, parameter_delta, token_count,
    validate_split_isolation, weighted_accumulation_windows,
)


class TemplateTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                            return_tensors):
        text = "".join(f"<{message['role']}>{message['content']}<end>\n"
                       for message in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(char) for char in text]


def record(task_id="train-task", source_group="train-group", split="train",
           target="return 7", prompt="fix function"):
    return {
        "sample_id": task_id, "task_id": task_id,
        "source_group": source_group, "split": split,
        "messages": [{"role": "user", "content": prompt}],
        "target": {"role": "assistant", "content": target},
        "provenance": {"strict_pass": True},
    }


def write_records(path, records):
    path.write_text("\n".join(json.dumps(item) for item in records) + "\n",
                    encoding="utf-8")
    return path


def test_template_native_ending_and_assistant_only_mask(tmp_path):
    data = SFTDataset(write_records(tmp_path / "train.jsonl", [record()]),
                      TemplateTokenizer())
    sample = data[0]
    supervised = "".join(chr(label) for label in sample["labels"] if label != -100)
    assert supervised == "return 7<end>\n"
    prefix = "".join(chr(token) for token, label in
                     zip(sample["input_ids"], sample["labels"]) if label == -100)
    assert prefix == "<user>fix function<end>\n<assistant>"


def test_incompatible_template_fails_closed(tmp_path):
    class IncompatibleTokenizer(TemplateTokenizer):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                                return_tensors):
            ids = super().apply_chat_template(messages, tokenize,
                                               add_generation_prompt, return_tensors)
            return [999] + ids if add_generation_prompt else ids

    path = write_records(tmp_path / "train.jsonl", [record()])
    with pytest.raises(ValueError, match="prefix mismatch"):
        SFTDataset(path, IncompatibleTokenizer())


def test_mapping_tokenizer_output_is_supported(tmp_path):
    class MappingTokenizer(TemplateTokenizer):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                                return_tensors):
            ids = super().apply_chat_template(messages, tokenize,
                                               add_generation_prompt, return_tensors)
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    data = SFTDataset(write_records(tmp_path / "train.jsonl", [record()]),
                      MappingTokenizer())
    assert "".join(chr(label) for label in data[0]["labels"] if label != -100) == "return 7<end>\n"


def test_positive_filter_dedup_and_invalid_schema(tmp_path):
    positive = record()
    duplicate = {**positive, "sample_id": "duplicate"}
    string_pass = {**record(target="wrong"), "provenance": {"strict_pass": "true"}}
    missing_group = {**record(target="missing"), "source_group": ""}
    invalid_target = {**record(), "target": {"role": "user", "content": "wrong"}}
    path = write_records(tmp_path / "train.jsonl",
                         [positive, duplicate, string_pass, missing_group, invalid_target])
    dataset = SFTDataset(path, TemplateTokenizer())
    assert len(dataset) == 1
    assert dataset.stats.skipped_duplicate == 1
    assert dataset.stats.skipped_fail == 1
    assert dataset.stats.skipped_invalid == 2
    assert len(dataset.stats.file_sha256) == 64


def test_split_mismatch_fails(tmp_path):
    path = write_records(tmp_path / "train.jsonl", [record(split="test")])
    with pytest.raises(ValueError, match="expected split=train"):
        SFTDataset(path, TemplateTokenizer())


@pytest.mark.parametrize("shared", ["task", "source_group", "prompt"])
def test_train_dev_isolation(tmp_path, shared):
    train_record = record()
    dev_record = record(task_id="dev-task", source_group="dev-group",
                        split="dev", prompt="another prompt")
    if shared == "task":
        dev_record["task_id"] = train_record["task_id"]
    elif shared == "source_group":
        dev_record["source_group"] = train_record["source_group"]
    else:
        dev_record["messages"] = train_record["messages"]
    train_data = SFTDataset(write_records(tmp_path / "train.jsonl", [train_record]),
                           TemplateTokenizer())
    dev_data = SFTDataset(write_records(tmp_path / "dev.jsonl", [dev_record]),
                         TemplateTokenizer(), split="dev")
    with pytest.raises(ValueError, match="泄漏"):
        validate_split_isolation(train_data.stats, dev_data.stats)


def test_independent_splits_and_padding(tmp_path):
    train_data = SFTDataset(write_records(tmp_path / "train.jsonl", [record()]),
                           TemplateTokenizer())
    dev_data = SFTDataset(write_records(tmp_path / "dev.jsonl", [record(
        task_id="dev", source_group="dev", split="dev", prompt="dev prompt")]),
        TemplateTokenizer(), split="dev")
    validate_split_isolation(train_data.stats, dev_data.stats)
    short = {"input_ids": [1, 2], "labels": [-100, 2]}
    long = {"input_ids": [1, 2, 3, 4], "labels": [-100, 2, 3, 4]}
    batch = Collator(0)([short, long])
    assert batch["labels"][0].tolist() == [-100, 2, -100, -100]
    assert batch["attention_mask"][0].tolist() == [1, 1, 0, 0]


def make_batch(targets):
    ids = torch.tensor([[0] + targets])
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "labels": torch.tensor([[-100] + targets])}


def test_token_weighted_gradient_and_tail_window():
    batches = [make_batch([1]), make_batch([3, 3, 3]), make_batch([2, 2])]
    windows = list(weighted_accumulation_windows(iter(batches), 2))
    assert [denominator for _, denominator in windows] == [4, 2]
    parameter = torch.tensor(0.5, requires_grad=True)
    for batch in windows[0][0]:
        targets = batch["labels"][batch["labels"] != -100].float()
        loss = ((parameter - targets) ** 2).mean()
        (loss * token_count(batch["labels"]) / windows[0][1]).backward()
    direct_parameter = torch.tensor(0.5, requires_grad=True)
    ((direct_parameter - torch.tensor([1., 3., 3., 3.])) ** 2).mean().backward()
    assert torch.allclose(parameter.grad, direct_parameter.grad)


class MeanTargetModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask, labels):
        return SimpleNamespace(loss=labels[labels != -100].float().mean())


def test_validation_loss_is_token_weighted_and_restores_mode():
    model = MeanTargetModel()
    model.train()
    result = evaluate_loss(model, [make_batch([1]), make_batch([3, 3, 3])],
                           torch.device("cpu"))
    assert result == {"loss": 2.5, "tokens": 4}
    assert model.training


def test_parameter_delta_counts_actual_changed_values():
    model = torch.nn.Linear(2, 1, bias=False)
    before = {"weight": model.weight.detach().clone()}
    with torch.no_grad():
        model.weight[0, 0].add_(1)
    result = parameter_delta(before, model)
    assert result["changed_parameter_tensors"] == 1
    assert result["changed_parameter_values"] == 1
    assert result["l2_delta"] == pytest.approx(1)


def test_cli_keeps_old_flags_and_requires_dev_before_hf_import(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_sft.py"
    spec = importlib.util.spec_from_file_location("production_train_sft", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.build_parser().parse_args(["--data", str(tmp_path / "train.jsonl")])
    assert args.lr == 2e-5
    assert args.epochs == 3
    with pytest.raises(ValueError, match="eval-data"):
        module.train(args)


@pytest.mark.parametrize("mapping_output", [False, True])
def test_evaluation_tokenizer_compatibility(mapping_output):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from evaluate import tokenize_prompt

    class TensorTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                                return_tensors):
            ids = torch.tensor([[1, 2, 3]])
            if mapping_output:
                return {"input_ids": ids, "attention_mask": torch.tensor([[0, 1, 1]])}
            return ids

    inputs, attention = tokenize_prompt(TensorTokenizer(), [], torch.device("cpu"))
    assert inputs.tolist() == [[1, 2, 3]]
    assert attention.tolist() == ([[0, 1, 1]] if mapping_output else [[1, 1, 1]])
