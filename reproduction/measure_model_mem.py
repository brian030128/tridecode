import torch
import torch.nn.functional as F
from transformers import LlamaTokenizer, cache_utils, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from transformers.cache_utils import DynamicCache
import time
import datasets
from datasets import load_dataset
import json


from origin import origin_generate, origin_warmup
from tree_decoding import tree_generate, tree_warmup
from run import run_bench_mark
from task import Task
from model_type import ModelType
from transformers import logging
from run import Metric
from typing import List
import os

logging.set_verbosity_error()

import sys

import GPUtil

os.environ['HF_HOME'] = '/work/u4320956/hf-cache'

def get_gpu_usage():
    gpus = GPUtil.getGPUs()
    total = 0
    for gpu in gpus:
        total += (torch.cuda.memory_allocated(device=gpu.id) / 1000000)
    return total



def name(type):
    match type:
        case ModelType.LLAMA3:
            return  "meta-llama/Llama-3.1-8B-Instruct"
        case ModelType.PHI35:
            return "microsoft/Phi-3.5-mini-instruct"
        case ModelType.MISTRAL:
            return "mistralai/Mistral-Small-24B-Instruct-2501"
        case ModelType.LLAMA3_70B:
            return "meta-llama/Llama-3.1-70B-Instruct"
    


def test_model(model_type:ModelType):
    tokenizer = AutoTokenizer.from_pretrained(name(model_type))
    model = AutoModelForCausalLM.from_pretrained(
        name(model_type),
        device_map="auto"
    )
    print(f"model {model_type.name} loaded took {get_gpu_usage()} MB")


test_model(ModelType.LLAMA3)
test_model(ModelType.PHI35)
test_model(ModelType.MISTRAL)
