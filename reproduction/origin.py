import torch
import gc as gpu_gc
from transformers import LlamaForCausalLM
from typing import Tuple, List
from transformers.models import metrics

def origin_warmup(model, tokenizer, prompt, num_beams, max_tokens):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    model.generate(input_ids, do_sample=False, num_beams=num_beams, max_new_tokens=max_tokens, temperature=None, top_p=None)
    

def origin_generate(model, tokenizer, prompt, num_beams, max_new_tokens, eos_token_id) -> Tuple[List[int], List[int], List[float]]:

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    outputs = model.generate(input_ids,do_sample=False, return_legacy_cache=False, num_beams=num_beams,
                             max_new_tokens=max_new_tokens, temperature=None, top_p = None, early_stopping=True)

    return (outputs[0][input_ids.shape[-1]:], metrics.memory_metrics, metrics.time_metrics)

def sampling_generate(model, tokenizer, prompt, num_beams, max_new_tokens, eos_token_id,temperature=1.0) -> Tuple[List[int], List[int], List[float]]:

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    outputs = model.generate(input_ids,do_sample=True, return_legacy_cache=False, temperature=temperature,
                             max_new_tokens=max_new_tokens, early_stopping=True)

    return (outputs[0][input_ids.shape[-1]:], metrics.memory_metrics, metrics.time_metrics)