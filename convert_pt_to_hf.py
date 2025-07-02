from typing import Literal
import torch
import safetensors.torch
import simple_parsing
import dataclasses
import transformers
transformers.LlamaForCausalLM


_PT_TO_HF_REPLACE_MAP = {
    ".attention.": ".self_attn.",
    ".feed_forward.": ".mlp.",
    ".w1.": ".gate_proj.",
    ".w2.": ".down_proj.",
    ".w3.": ".up_proj.",
    ".wk.": ".k_proj.",
    ".wq.": ".q_proj.",
    ".wv.": ".v_proj.",
    ".wo.": ".o_proj.",
    ".attention_norm.": ".input_layernorm.",
    ".ffn_norm.": ".post_attention_layernorm.",
    "tok_embeddings.": "embed_tokens.",
    "output.": "lm_head.",
}


def pt_to_hf_key_name(key: str) -> str:
    for k, v in _PT_TO_HF_REPLACE_MAP.items():
        key = key.replace(k, v)
    if "lm_head" in key:
        return key
    return "model." + key


def apply_scales(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return (
        torch.repeat_interleave(
            scales.float(), weight.shape[1] // scales.shape[1], dim=1
        )
        * weight.float()
    )


def permute(w, n_heads, dim1, dim2):
    if w is None:
        return None
    return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def pt_to_hf_state_dict(
    pt_state_dict: dict[str, torch.Tensor], dtype: torch.dtype = torch.bfloat16
) -> dict[str, torch.Tensor]:
    """
    Convert a state dict for the original implementation of Llama and SpinQuant
    to a state dict for the Hugging Face implementation.
    If the state dict contains "scales" (i.e. quantized by SpinQuant for ExecuTorch)
    it applies the grouped scales to the weights as well.

    For implementation details, see:
    https://github.com/meta-llama/llama3/blob/main/llama/model.py
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    """
    dim = 2048
    n_heads = 32
    dims_per_head = dim // n_heads
    num_key_value_heads = 8
    key_value_dim = dims_per_head * num_key_value_heads
    n_layers = 16

    def get_weight(pt_state_dict, key, do_permute=False, is_query=False):
        weight = pt_state_dict[key + ".weight"]
        scale = pt_state_dict.get(key + ".scale", None)
        if scale is not None:
            weight = apply_scales(weight, scale)
        if do_permute:
            if is_query:
                weight = permute(weight, n_heads, dim, dim)
                # scale = permute(scale, n_heads, dim, dim)
            else:
                weight = permute(weight, num_key_value_heads, key_value_dim, dim)
                # scale = permute(scale, num_key_value_heads, key_value_dim, dim)
        return weight

    hf_weight = {
        "model.embed_tokens.weight": get_weight(pt_state_dict, "tok_embeddings"),
        "model.norm.weight": get_weight(pt_state_dict, "norm"),
        "lm_head.weight": get_weight(pt_state_dict, "output"),
    }
    for layer_i in range(n_layers):
        hf_weight[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = get_weight(
            pt_state_dict, f"layers.{layer_i}.attention.wq", do_permute=True, is_query=True
        )
        hf_weight[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = get_weight(
            pt_state_dict, f"layers.{layer_i}.attention.wk", do_permute=True, is_query=False
        )
        hf_weight[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = get_weight(
            pt_state_dict, f"layers.{layer_i}.attention.wv"
        )
        hf_weight[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = get_weight(
            pt_state_dict, f"layers.{layer_i}.attention.wo"
        )
        hf_weight[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = get_weight(
            pt_state_dict, f"layers.{layer_i}.feed_forward.w1"
        )
        hf_weight[f"model.layers.{layer_i}.mlp.down_proj.weight"] = get_weight(
            pt_state_dict, f"layers.{layer_i}.feed_forward.w2"
        )
        hf_weight[f"model.layers.{layer_i}.mlp.up_proj.weight"] = get_weight(
            pt_state_dict, f"layers.{layer_i}.feed_forward.w3"
        )
        hf_weight[f"model.layers.{layer_i}.input_layernorm.weight"] = get_weight(
            pt_state_dict, f"layers.{layer_i}.attention_norm"
        )
        hf_weight[f"model.layers.{layer_i}.post_attention_layernorm.weight"] = get_weight(
            pt_state_dict, f"layers.{layer_i}.ffn_norm"
        )
    # for k, v in pt_state_dict.items():
    #     if "weight" in k:
    #         weight = v
    #         scale_key = k.replace("weight", "scales")
    #         if ".wk." in k:
    #             weight = permute(weight, num_key_value_heads, key_value_dim, dim)
    #         elif ".wq." in k:
    #             weight = permute(weight, n_heads, dim, dim)
    #         if scale_key in pt_state_dict:
    #             weight = apply_scales(weight, pt_state_dict[scale_key])
    #         hf_weight[pt_to_hf_key_name(k)] = weight.to(dtype)
    #     elif "scale" not in k:
            # raise ValueError(f"Unknown key: {k}")
    return hf_weight


@dataclasses.dataclass
class ConvertPTToHFArgs:
    pt_model_path: str = simple_parsing.field(alias="-m")
    hf_model_path: str = simple_parsing.field(alias="-o")
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"

    def __post_init__(self):
        self.torch_dtype = getattr(torch, self.dtype)


def main(args: ConvertPTToHFArgs) -> None:
    pt_state_dict = torch.load(args.pt_model_path, map_location="cpu")
    hf_state_dict = pt_to_hf_state_dict(pt_state_dict, args.torch_dtype)
    # del hf_state_dict["lm_head.weight"]
    safetensors.torch.save_file(hf_state_dict, args.hf_model_path, metadata={"format": "pt"})


if __name__ == "__main__":
    main(args=simple_parsing.parse(ConvertPTToHFArgs))
