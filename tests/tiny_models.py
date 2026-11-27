# -*- coding: utf-8 -*-
"""测试用的微型真实架构：结构、注意力的种类和正式模型一致，只是很小，能在 CPU 上跑。"""
import torch
import transformers


def tiny_qwen35(dtype=torch.bfloat16):
    """4 层，每 4 层有 3 层是线性注意力，和 Qwen3.5 一样。"""
    config = transformers.Qwen3_5TextConfig(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, vocab_size=512,
        linear_key_head_dim=16, linear_num_key_heads=2,
        linear_value_head_dim=16, linear_num_value_heads=2,
        max_position_embeddings=4096)
    torch.manual_seed(0)
    return transformers.Qwen3_5ForCausalLM(config).to(dtype)


def tiny_qwen25(dtype=torch.bfloat16):
    config = transformers.Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, vocab_size=512)
    torch.manual_seed(0)
    return transformers.Qwen2ForCausalLM(config).to(dtype)
