import dataclasses
import datetime
import torch
import transformers

import simple_parsing

from eval_utils import rotation_utils
from utils import fuse_norm_utils


@dataclasses.dataclass
class RotateModelArgs:
    model: str = simple_parsing.field(alias="-m")
    optimized_rotation_path: str = simple_parsing.field(alias="-o")
    save_path: str = simple_parsing.field(alias="-s")
    rotate_mode: str = "random"
    apply_r4: bool = False
    bf16: bool = False
    seed: int = 0


def rotate_and_save(args: RotateModelArgs):
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if model.config.tie_word_embeddings:
        model.config.tie_word_embeddings = False
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    transformers.set_seed(args.seed)
    fuse_norm_utils.fuse_layer_norms(model)
    rotation_utils.rotate_model(model, args)
    for param in model.parameters():
        param.data = param.contiguous()
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    rotate_and_save(simple_parsing.parse(RotateModelArgs))
