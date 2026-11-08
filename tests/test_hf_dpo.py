# -*- coding: utf-8 -*-
"""hf_dpo: dataset masking, collator, and a micro policy/reference step."""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentlab.dpo.loss import dpo_loss, sequence_logp
from agentlab.training.hf_dpo import DPOCollator, DPODataset, _logp_batch, _target_only


class FakeChatTokenizer:
    """Deterministic chat template: prefix rendering makes target a suffix."""
    pad_token = "<pad>"
    pad_token_id = 0

    def _render(self, messages):
        text = ""
        for message in messages:
            text += f"<|{message['role']}|>{message['content']}"
        return text

    def apply_chat_template(self, messages, tokenize=True,
                            add_generation_prompt=True, return_tensors=None):
        text = self._render(messages)
        if add_generation_prompt:
            text += "<|assistant|>"
        assert tokenize
        return [ord(ch) % 250 + 2 for ch in text]


def _pair(pair_id="p1", task_id="t1", chosen="good()", rejected="bad(",
          split="train"):
    return {
        "pair_id": pair_id, "task_id": task_id, "source_group": "g1",
        "split": split,
        "messages": [{"role": "user", "content": "fix it"}],
        "chosen": {"role": "assistant", "content": chosen},
        "rejected": {"role": "assistant", "content": rejected},
    }


def _write(tmp_path, records, name="prefs.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                            for r in records), encoding="utf-8")
    return path


def test_target_only_supervises_suffix_only():
    tokenizer = FakeChatTokenizer()
    messages = [{"role": "user", "content": "hi"}]
    full, labels = _target_only(tokenizer, messages,
                                {"role": "assistant", "content": "ok"})
    prefix, _ = _target_only.__wrapped__(tokenizer, messages, None) \
        if hasattr(_target_only, "__wrapped__") else (None, None)
    # prefix = template(messages, add_generation_prompt=True)
    prefix_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True)
    assert full[:len(prefix_ids)] == prefix_ids
    assert labels[:len(prefix_ids)] == [-100] * len(prefix_ids)
    assert labels[len(prefix_ids):] == full[len(prefix_ids):]
    assert sum(label != -100 for label in labels) > 0


def test_dataset_filters_and_masks(tmp_path):
    records = [
        _pair("p1", "t1"),
        _pair("p1", "t1"),                      # duplicate → skipped
        _pair("p2", "t2", split="dev"),         # wrong split → skipped
        _pair("p3", "t3", chosen=""),           # empty chosen target → 0 sup → skipped
    ]
    data = DPODataset(_write(tmp_path, records), FakeChatTokenizer(), 2048,
                      split="train")
    assert len(data) == 1
    assert data.stats.total == 4
    assert data.stats.skipped_duplicate == 1
    assert data.stats.skipped_invalid == 2
    sample = data[0]
    prefix_len = len(FakeChatTokenizer().apply_chat_template(
        sample_messages := [{"role": "user", "content": "fix it"}],
        tokenize=True, add_generation_prompt=True))
    assert sample["labels_chosen"][:prefix_len] == [-100] * prefix_len
    assert sample["labels_rejected"][:prefix_len] == [-100] * prefix_len
    # chosen 与 rejected 共享完全一致的前缀（共同状态）
    assert sample["input_ids_chosen"][:prefix_len] == \
        sample["input_ids_rejected"][:prefix_len]


def test_collator_pads_both_sides_and_keeps_masks():
    tokenizer = FakeChatTokenizer()
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "abc"}], add_generation_prompt=True)
    batch = [
        {"input_ids_chosen": ids + [7, 8], "labels_chosen": [-100] * len(ids) + [7, 8],
         "input_ids_rejected": ids + [9], "labels_rejected": [-100] * len(ids) + [9]},
        {"input_ids_chosen": ids + [5], "labels_chosen": [-100] * len(ids) + [5],
         "input_ids_rejected": ids + [6, 6], "labels_rejected": [-100] * len(ids) + [6, 6]},
    ]
    out = DPOCollator(pad_token_id=0)(batch)
    width = out["input_ids_chosen"].shape[1]
    assert width == len(ids) + 2
    assert out["input_ids_chosen"][1][width - 1].item() == 0   # padding
    assert out["labels_chosen"][1][width - 1].item() == -100
    assert out["attention_chosen"][1][width - 1].item() == 0
    assert (out["labels_rejected"][:, :len(ids)] == -100).all()
    assert (out["attention_rejected"][:, :len(ids)] == 1).all()


class TinyLM(torch.nn.Module):
    vocab = 32

    def __init__(self, scale=1.0):
        super().__init__()
        torch.manual_seed(7)
        self.emb = torch.nn.Embedding(self.vocab, 16)
        self.head = torch.nn.Linear(16, self.vocab)
        self.scale = scale

    def forward(self, input_ids, attention_mask=None):
        return type("O", (), {"logits": self.head(self.emb(input_ids)) * self.scale})()


def test_micro_dpo_step_moves_policy_toward_chosen():
    ids_c = torch.tensor([[1, 2, 3, 4, 5]])
    labels_c = torch.tensor([[-100, -100, 3, 4, 5]])
    ids_r = torch.tensor([[1, 2, 9, 8, 7]])
    labels_r = torch.tensor([[-100, -100, 9, 8, 7]])
    att = torch.ones_like(ids_c)
    policy, reference = TinyLM(), TinyLM()
    reference.load_state_dict(policy.state_dict())
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    device = torch.device("cpu")
    before = policy.head.weight.detach().clone()
    pol_c = _logp_batch(policy, ids_c, labels_c, att, device, False)[0]
    pol_r = _logp_batch(policy, ids_r, labels_r, att, device, False)[0]
    with torch.no_grad():
        ref_c = _logp_batch(reference, ids_c, labels_c, att, device, False)[0]
        ref_r = _logp_batch(reference, ids_r, labels_r, att, device, False)[0]
    metrics = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1)
    assert metrics["loss"].requires_grad
    metrics["loss"].backward()
    assert policy.head.weight.grad is not None
    assert policy.head.weight.grad.abs().sum() > 0
    assert torch.isfinite(policy.head.weight.grad).all()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.5)
    optimizer.step()
    assert not torch.equal(before, policy.head.weight.detach())
    # reference 不得有梯度
    assert all(p.grad is None for p in reference.parameters())


def test_adaptive_beta_dataset_reads_group_solvable(tmp_path):
    """DPODataset passes group_solvable through for adaptive β."""
    records = [
        _pair("p1", "t1", chosen="a()", rejected="b(", group_solvable=False),
        _pair("p2", "t2", chosen="c()", rejected="d(", group_solvable=True),
        _pair("p3", "t3", chosen="e()", rejected="f("),  # legacy
    ]
    data = DPODataset(_write(tmp_path, records), FakeChatTokenizer(), 2048, "train")
    assert len(data) == 3
    assert data[0]["group_solvable"] is False
    assert data[1]["group_solvable"] is True
    assert data[2]["group_solvable"] is None


def _pair(pair_id="p1", task_id="t1", chosen="good()", rejected="bad(",
          split="train", group_solvable=None):
    d = {
        "pair_id": pair_id, "task_id": task_id, "source_group": "g1",
        "split": split,
        "messages": [{"role": "user", "content": "fix it"}],
        "chosen": {"role": "assistant", "content": chosen},
        "rejected": {"role": "assistant", "content": rejected},
    }
    if group_solvable is not None:
        d["group_solvable"] = group_solvable
    return d


def test_adaptive_beta_collator_passes_mask():
    """Collator converts group_solvable to a long tensor."""
    tokenizer = FakeChatTokenizer()
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "q"}], add_generation_prompt=True)
    batch = [
        {"input_ids_chosen": ids + [1], "labels_chosen": [-100] * len(ids) + [1],
         "input_ids_rejected": ids + [2], "labels_rejected": [-100] * len(ids) + [2],
         "group_solvable": False},
        {"input_ids_chosen": ids + [3], "labels_chosen": [-100] * len(ids) + [3],
         "input_ids_rejected": ids + [4], "labels_rejected": [-100] * len(ids) + [4],
         "group_solvable": True},
    ]
    out = DPOCollator(pad_token_id=0)(batch)
    assert out["group_solvable"].tolist() == [0, 1]
