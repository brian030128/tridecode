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

os.environ['HF_HOME'] = '/work/u4320956/hf-cache'
sys.setrecursionlimit(3000)





def run_task(model_type, model, tokenizer ,task: Task, data_num: range, tree_params, origin_params):
    tree_warmup(model, tokenizer, "This is a test", 3, 1000,  [ model.config.eos_token_id ])

    ds = task.get_ds()
    for parameter in tree_params:
        if parameter[0] == 1:
            continue

        path = f"out/{model_type.name}/tree/{task.type().name}"
        os.makedirs(path, exist_ok=True)
        print("processing tree ",parameter[0], "_",parameter[1] )
        with open(f"{path}/{parameter[0]}_{parameter[1]}.jsonl", "w") as out_file:
            metrics = run_bench_mark(model, tokenizer, ds.select(data_num), tree_generate, task, model_type, parameter[0], parameter[1])
            for metric in metrics:
                out_file.write(json.dumps(metric.to_dict()) + "\n")

    origin_warmup(model, tokenizer, "This is a test", 3, 1000)

    for parameter in origin_params:
        path = f"out/{model_type.name}/origin/{task.type().name}"
        os.makedirs(path, exist_ok=True)
        print("processing origin ",parameter[0], "_",parameter[1] )
        with open(f"{path}/{parameter[0]}_{parameter[1]}.jsonl", "w") as out_file:
            metrics = run_bench_mark(model, tokenizer, ds.select(data_num), origin_generate, task, model_type, parameter[0], parameter[1])
            for metric in metrics:
                out_file.write(json.dumps(metric.to_dict()) + "\n")



def name(type):
    match type:
        case ModelType.LLAMA3:
            return  "meta-llama/Llama-3.1-8B-Instruct"
        case ModelType.PHI35:
            return "microsoft/Phi-3.5-mini-instruct"
        case ModelType.MISTRAL:
            return "mistralai/Mistral-Small-24B-Instruct-2501"
    


def test_model(model_type:ModelType, tree_params, origin_params):
    tokenizer = AutoTokenizer.from_pretrained(name(model_type))
    model = AutoModelForCausalLM.from_pretrained(
        name(model_type),
        device_map="auto"
    )

    from task import HumanEvalTask, Gsm8kTask,CNNSumTask, WMTTransTask
    #run_task(model_type,model,tokenizer,Gsm8kTask(), range(100), tree_params, origin_params)
    #run_task(model_type,model,tokenizer,HumanEvalTask(),range(164), tree_params, origin_params)
    #run_task(model_type,model,tokenizer,CNNSumTask(),range(100), tree_params, origin_params)
    run_task(model_type,model,tokenizer,WMTTransTask(),range(100), tree_params, origin_params)


# beams / max_tokens
parameters = [
    (1, 1000),
    (3, 1000),
    (9 , 1000)
]

test_model(ModelType.PHI35, parameters, parameters)
#test_model(ModelType.LLAMA3, parameters, parameters)
# test_model(ModelType.MISTRAL, 
#            [(3, 1000), (6, 1000)],
#            [(1, 1000),(3, 1000), (6, 1000)])


