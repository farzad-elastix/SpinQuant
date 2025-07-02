import torch

import convert_pt_to_hf


PT_TO_HF_NAMES = {
    "layers.0.feed_forward.w1.weight": "model.layers.0.mlp.gate_proj.weight",
    "layers.0.feed_forward.w2.weight": "model.layers.0.mlp.down_proj.weight",
    "layers.0.feed_forward.w3.weight": "model.layers.0.mlp.up_proj.weight",
    "layers.0.attention.wq.weight": "model.layers.0.self_attn.q_proj.weight",
    "layers.0.attention.wk.weight": "model.layers.0.self_attn.k_proj.weight",
    "layers.0.attention.wv.weight": "model.layers.0.self_attn.v_proj.weight",
    "layers.0.attention.wo.weight": "model.layers.0.self_attn.o_proj.weight",
    "layers.0.attention_norm.weight": "model.layers.0.input_layernorm.weight",
    "layers.0.ffn_norm.weight": "model.layers.0.post_attention_layernorm.weight",
    "tok_embeddings.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}


def test_pt_to_hf_state_dict():
    for k, v in PT_TO_HF_NAMES.items():
        assert convert_pt_to_hf.pt_to_hf_key_name(k) == v


def test_apply_scales():
    weight = torch.tensor([[1, 2, 3, 4], [3, 4, 5, 6]])
    scales = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected = torch.tensor([[1, 2, 6, 8], [9, 12, 20, 24]]).float()
    assert torch.allclose(convert_pt_to_hf.apply_scales(weight, scales), expected)


# Example PT weights
# norm.weight: torch.Size([2048])
# tok_embeddings.weight: torch.Size([128256, 2048])
# output.weight: torch.Size([128256, 2048])
# tok_embeddings.scales: torch.Size([128256, 1])
# output.scales: torch.Size([128256, 1])
# layers.0.attention_norm.weight: torch.Size([2048])
# layers.0.ffn_norm.weight: torch.Size([2048])
# layers.0.attention.wq.weight: torch.Size([2048, 2048])
# layers.0.attention.wk.weight: torch.Size([512, 2048])
# layers.0.attention.wv.weight: torch.Size([512, 2048])
# layers.0.attention.wo.weight: torch.Size([2048, 2048])
# layers.0.feed_forward.w1.weight: torch.Size([8192, 2048])
# layers.0.feed_forward.w2.weight: torch.Size([2048, 8192])
# layers.0.feed_forward.w3.weight: torch.Size([8192, 2048])

# Example HF weights
# model.embed_tokens.weight: torch.Size([128256, 2048])
# model.norm.weight: torch.Size([2048])
# model.layers.0.self_attn.q_proj.weight: torch.Size([2048, 2048])
# model.layers.0.self_attn.k_proj.weight: torch.Size([512, 2048])
# model.layers.0.self_attn.v_proj.weight: torch.Size([512, 2048])
# model.layers.0.self_attn.o_proj.weight: torch.Size([2048, 2048])
# model.layers.0.mlp.gate_proj.weight: torch.Size([8192, 2048])
# model.layers.0.mlp.up_proj.weight: torch.Size([8192, 2048])
# model.layers.0.mlp.down_proj.weight: torch.Size([2048, 8192])
# model.layers.0.input_layernorm.weight: torch.Size([2048])
# model.layers.0.post_attention_layernorm.weight: torch.Size([2048])
