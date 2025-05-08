from typing import Callable, Tuple, List
from transformers import LlamaForCausalLM, LlamaTokenizer
import datasets
import time
from task import Task, TaskType
from tqdm import tqdm
import torch
import gc as gpu_gc

from enum import Enum
from model_type import ModelType

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

    def __init__(self, id: str, answer: str,model_memory: int, time_taken: float, memory_usage: List[int], time_metric: List[float], score: float, input_len: int,output_len: int, output: str):
        self.model_memory = model_memory
        self.input_kv_memory = memory_usage[0]
        self.id = id
        self.answer = answer
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
    model,
    tokenizer,
    dataset: datasets.Dataset,
    generate: Callable[[LlamaForCausalLM, LlamaTokenizer, str, int, int, List[int]], Tuple[str, List[int]]],
    task: Task,
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
        prompt = task.get_prompt(model_type, data['text'])

        torch.cuda.empty_cache()
        gpu_gc.collect()
        metrics.clear()
        
        # Update progress bar description with current sample ID
        
        model_memory = get_gpu_usage()

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        if input_ids.shape[1] + max_new_tokens > 6000:
            continue
        start = time.time()
        try:
            if model_type == ModelType.LLAMA3:
                output, memory_usage, time_metric = generate(model, tokenizer, prompt, num_beams, max_new_tokens, [model.config.eos_token_id])
            elif model_type == ModelType.MISTRAL:
                output, memory_usage, time_metric = generate(model, tokenizer, prompt, num_beams, max_new_tokens, [model.config.eos_token_id])
            elif model_type == ModelType.PHI35:
                output, memory_usage, time_metric = generate(model, tokenizer, prompt, num_beams, max_new_tokens,  [32007, 32001, 32000] )
        except NotImplementedError:
            print("err")
            continue
        completion = tokenizer.decode(output, skip_special_tokens=True)
        completion = task.extract_answer(completion)

        end = time.time()
        print(":", completion)

        score = 0        
        
        metric = Metric(
            id= data['id'] if 'id' in data.keys() else "",
            answer=data['answer'] if 'answer' in data.keys() else "",
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

