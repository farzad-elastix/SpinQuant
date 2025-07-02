import dataclasses
import gc
import simple_parsing
import torch

from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.quantization import FORMAT
from logbar import LogBar


log = LogBar.shared()


@dataclasses.dataclass
class QuantArgs:
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    save_path: str = ""
    cfg_bits: int = 4
    cfg_groupsize: int = 32
    cfg_v2: bool = True
    inputs_max_length: int = 2048 # in tokens
    
    def __post_init__(self):
        if not self.save_path:
            self.save_path = f"gptq_v2_{self.cfg_v2}_bit_{self.cfg_bits}_gpsize_{self.cfg_groupsize}_{self.model_id.split('/')[-1]}"

def get_calib_data(tokenizer, rows: int, max_length: int):
    calibration_dataset = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00000-of-01024.json.gz",
        split="train"
    )
    datas = []
    for index, sample in enumerate(calibration_dataset):
        tokenized = tokenizer(sample["text"])
        if len(tokenized.data['input_ids']) <= max_length:
            datas.append(tokenized)
            if len(datas) >= rows:
                break
    return datas

args = simple_parsing.parse(QuantArgs)
quant_config = QuantizeConfig(
    bits=args.cfg_bits,
    group_size=args.cfg_groupsize,
    format=FORMAT.GPTQ,
    desc_act=True,
    sym=True,
    v2=args.cfg_v2,
)

log.info(f"QuantConfig: {quant_config}")
log.info(f"Save Path: {args.save_path}")

# load un-quantized native model
model = GPTQModel.load(args.model_id, quant_config)

# load calibration data
calibration_dataset = get_calib_data(tokenizer=model.tokenizer, rows=256, max_length=args.inputs_max_length)

model.quantize(calibration_dataset, batch_size=1)

model.save(args.save_path)
log.info(f"Quant Model Saved to: {args.save_path}")

del model
gc.collect()
torch.cuda.empty_cache()
