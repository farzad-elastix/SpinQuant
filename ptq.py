# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import datetime
from logging import Logger

import torch
import torch.distributed as dist
from transformers import LlamaTokenizerFast
import transformers
from eval_utils.main import ptq_model
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import data_utils, eval_utils, utils
from utils import quant_utils
from utils.process_args import process_args_ptq

log: Logger = utils.get_logger("spinquant")


def train() -> None:
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = utils.get_local_rank()

    log.info("the rank is {}".format(local_rank))

    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    model.cuda()

    model = ptq_model(ptq_args, model, model_args)
    model.seqlen = training_args.model_max_length
    if local_rank == 0:
        log.info("Model PTQ completed {}".format(model))
        log.info("Start to load tokenizer...")
    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
    )
    log.info("Complete tokenizer loading...")
    model.config.use_cache = False

    testloader = data_utils.get_wikitext2(
        seed=ptq_args.seed,
        seqlen=2048,
        tokenizer=tokenizer,
        eval_mode=True,
    )

    dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, ptq_args)
    log.info("wiki2 ppl is: {}".format(dataset_ppl))

    import pdb

    pdb.set_trace()

    layers = quant_utils.find_qlayers(model)
    for path, layer in layers.items():
        parent = model
        *parent_path, final = path.split('.')
        for k in parent_path:
            parent = getattr(parent, k)
        setattr(parent, final, layer.module)

    model.save_pretrained(os.path.join(os.path.dirname(args.save_qmodel_path), "ptq"))



if __name__ == "__main__":
    train()
