from typing import Callable, Tuple, List
from transformers import LlamaForCausalLM, LlamaTokenizer
import datasets
import time
from tqdm import tqdm
import torch
import gc as gpu_gc

from enum import Enum

class TaskType(Enum):
    SUM = 1
    HUMAN_EVAL = 2
    QASPER = 3
    QSUM = 4

class ModelType(Enum):
    LLAMA2 = 1
    PHI35 = 2


from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex
from pynvml import nvmlDeviceGetMemoryInfo, nvmlShutdown
from functools import lru_cache

from rouge_score import rouge_scorer


import GPUtil

def get_gpu_usage():
    gpus = GPUtil.getGPUs()
    total = 0
    for gpu in gpus:
        total += gpu.memoryUsed
    return total



class Metric:

    def __init__(self, id: str,model_memory: int, time_taken: float, memory_usage: List[int], time_metric: List[float], score: float, input_len: int,output_len: int, output: str):
        self.model_memory = model_memory
        self.input_kv_memory = memory_usage[0]
        self.id = id
        self.time_taken = time_taken
        self.memory_usage = memory_usage
        self.time_metric = time_metric
        self.score = score
        self.input_len = input_len
        self.output_len = output_len
        self.output = output

    def to_dict(self):
        return {
            "id": self.id,
            "model_memory": self.model_memory,
            "time_taken": self.time_taken,
            "input_kv_memory": self.input_kv_memory,
            "memory_usage": self.memory_usage,
            "time_metric": self.time_metric,
            "score": self.score,
            "input_len": self.input_len,
            "output_len": self.output_len,
            "output": self.output
        }
    
from transformers.models import metrics

def run_bench_mark(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    dataset: datasets.Dataset,
    generate: Callable[[LlamaForCausalLM, LlamaTokenizer, str, int, int, List[int]], Tuple[str, List[int]]],
    task_type: TaskType,
    model_type: ModelType,
    num_beams = 10,
    max_new_tokens = 1000,
) -> List[Metric]:
    
    # Create tqdm progress bar
    progress_bar = tqdm(
        range(len(dataset)),
        desc="Running benchmark",
        unit="sample",
        ncols=100,
        position=0,
        leave=True
    )

    torch.cuda.empty_cache()
    gpu_gc.collect()
    metrics.clear()


    metrics_list = []
    
    for i in progress_bar:
        data = dataset[i]
        if task_type == TaskType.SUM:
            if model_type == ModelType.LLAMA2:
                prompt = f"""<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

Output summary directly.
Article:
{data['text']}
Summary: [/INST]"""
            elif model_type == ModelType.PHI35:
                prompt = f"""<|system|>
You are a helpful assistant.<|end|>
<|user|>
Output summary directly.
{data['text']}<|end|>
<|assistant|>"""
        elif task_type == TaskType.HUMAN_EVAL:
            if model_type == ModelType.LLAMA2:
                prompt = f"""<s>[INST] <<SYS>>
You are a programmer.
<</SYS>>
Complete the following function. No explaination is needed, output the code directly.
{data['text']} [/INST]"""
            elif model_type == ModelType.PHI35:
                prompt = f"""<|system|>
You are a programmer.<|end|>
<|user|>
Complete the following function. No explaination is needed, output the code directly.
{data['text']}<|end|>
<|assistant|>"""
        elif task_type == TaskType.QASPER:
            if model_type == ModelType.PHI35:
                prompt = f"""<|system|>
You are a helpful assistant.<|end|>
<|user|>{data['text']}<|end|>
<|assistant|>"""
        elif task_type == TaskType.QSUM:
            if model_type == ModelType.PHI35:
                prompt = f"""<|system|>
You are a helpful assistant.<|end|>
<|user|>{data['text']}<|end|>
<|assistant|>"""
        torch.cuda.empty_cache()
        gpu_gc.collect()
        metrics.clear()
        # Update progress bar description with current sample ID
        progress_bar.set_description(f"Processing sample {data['id']}")
        
        model_memory = get_gpu_usage()

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        if input_ids.shape[1] + max_new_tokens > 6000:
            continue
        start = time.time()
        try:
            if model_type == ModelType.LLAMA2:
                output, memory_usage, time_metric = generate(model, tokenizer, prompt, num_beams, max_new_tokens, [ model.config.eos_token_id ])
            elif model_type == ModelType.PHI35:
                output, memory_usage, time_metric = generate(model, tokenizer, prompt, num_beams, max_new_tokens,  [32007, 32001, 32000] )
        except NotImplementedError:
            #This version of huggingface may produce the exception
            #Make sure that a `_reorder_cache` function is correctly implemented in transformers.models
            #We ignore it for now, and hope it'll be fixed in the future
            print("err")
            continue
        completion = tokenizer.decode(output, skip_special_tokens=True)
        end = time.time()
        print(":", completion)

        score = 0
        if task_type == TaskType.SUM or task_type == TaskType.QASPER or task_type == TaskType.QSUM:
            rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            score = rouge.score(completion, data['answer'])['rougeL'].fmeasure
        
        
        metric = Metric(
            id=data['id'],
            model_memory=model_memory,
            time_taken=end - start,
            memory_usage=memory_usage,
            time_metric=time_metric,
            score=score,
            input_len=input_ids.shape[1],
            output_len=len(output),
            output=completion
        )
        metrics_list.append(metric)

        # Update progress bar postfix with current metrics
        progress_bar.set_postfix({
            'time': f"{metric.time_taken:.2f}s",
            'mem': f"{max(memory_usage) if memory_usage else 0:.2f}MB"
        })

        #del output        
    
    return metrics_list

