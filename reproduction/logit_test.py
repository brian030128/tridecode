import argparse
import json
import os
import random
from typing import List

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from reproduction.tree_decoding import SearchNode, SearchTree, generate_causal_mask, gc

# Available model shortcuts used by other experiments
MODEL_CHOICES = {
    "llama3": "meta-llama/Llama-3.1-8B-Instruct",
    "phi35": "microsoft/Phi-3.5-mini-instruct",
    "mistral": "mistralai/Mistral-Small-24B-Instruct-2501",
    "llama3_70b": "meta-llama/Llama-3.1-70B-Instruct",
}

# Datasets used in the reproduction experiments
DATASET_CHOICES = {
    "human_eval": {"path": "openai_humaneval", "config": None, "split": "test", "text_column": "prompt"},
    "gsm8k": {"path": "openai/gsm8k", "config": "main", "split": "test", "text_column": "question"},
    "cnn": {"path": "abisee/cnn_dailymail", "config": "3.0.0", "split": "train+validation+test", "text_column": "article"},
    "wmt": {"path": "wmt/wmt_t2t", "config": None, "split": "train", "text_column": "translation"},
}
from transformers.cache_utils import DynamicCache


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def record_baseline_logits(
    model,
    tokenizer,
    prompt: str,
    beam_width: int,
    max_new_tokens: int,
    eos_token_id: List[int],
):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            inputs.input_ids,
            do_sample=False,
            num_beams=beam_width,
            max_new_tokens=max_new_tokens,
            temperature=None,
            top_p=None,
            early_stopping=False,
            return_dict_in_generate=True,
            output_scores=True,
            output_attentions=False,
            output_hidden_states=False,
        )

    logits = [F.log_softmax(score, dim=-1).cpu() for score in out.scores]

    # Reconstruct beam tokens/parents from logits to capture the full beam tree
    vocab_size = logits[0].shape[-1]
    steps = []
    eos = set(eos_token_id)
    beam_scores = torch.full((beam_width,), float("-inf"))
    beam_scores[0] = 0.0

    first_scores = logits[0][0]
    topk_scores, topk_tokens = torch.topk(first_scores, beam_width, dim=-1)
    steps.append({"tokens": topk_tokens.tolist(), "parents": [-1] * beam_width})
    beam_scores = topk_scores

    for log_prob in logits[1:]:
        scores = log_prob + beam_scores.view(-1, 1)
        flat_scores = scores.view(-1)
        topk_scores, topk_ids = torch.topk(flat_scores, beam_width, dim=0, largest=True, sorted=True)
        next_beams = (topk_ids // vocab_size).to(torch.long)
        next_tokens = topk_ids % vocab_size
        steps.append({"tokens": next_tokens.tolist(), "parents": next_beams.tolist()})
        beam_scores = topk_scores
        eos_mask = torch.tensor([t.item() in eos for t in next_tokens], dtype=torch.bool)
        beam_scores = beam_scores.masked_fill(eos_mask, float("-inf"))
        if eos_mask.all():
            break

    logits = logits[: len(steps)]
    return logits, steps


def record_trie_logits(
    model, tokenizer, prompt: str, beam_width: int,
    max_new_tokens: int, eos_token_id: List[int]
):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    logits_list = []
    tree_steps = []
    past_key_values = DynamicCache()
    with torch.no_grad():
        outputs = model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            num_logits_to_keep=1,
        )
        log_probs = F.log_softmax(outputs.logits[:, -1, :].float(), dim=-1)
        # Expand the initial logits to ``beam_width`` rows so that the first step
        # matches the shape produced by standard beam search.
        logits_list.append(log_probs.expand(beam_width, -1).cpu())
        past_key_values = outputs.past_key_values
        token_scores, tokens = torch.topk(
            log_probs,
            beam_width,
            dim=-1,
            largest=True,
            sorted=True,
        )
        searchTree = SearchTree(model, beam_width=beam_width)
        newest_branch = []
        idx = 0
        for i in range(beam_width):
            node = SearchNode(searchTree, idx, tokens[0][i], token_scores[0][i])
            idx += 1
            newest_branch.append(node)
            searchTree.root.append(node)
            searchTree.node_count += 1
        tree_steps.append({"tokens": [t.item() for t in tokens[0]], "parents": [-1] * beam_width})
        for step in range(input_ids.shape[1], max_new_tokens + input_ids.shape[1]):
            position_ids = torch.tensor([[step for _ in range(beam_width)]], device=model.device)
            attention_mask = generate_causal_mask(searchTree, input_ids.shape[1], newest_branch).to(model.device)
            step_input_ids = torch.tensor([[node.token_id for node in newest_branch]], device=model.device)
            outputs = model(step_input_ids, past_key_values=past_key_values,
                            position_ids=position_ids, attention_mask=attention_mask, use_cache=True)
            past_key_values = outputs.past_key_values
            log_probs = F.log_softmax(outputs.logits, dim=-1)
            logits_list.append(log_probs[0].cpu())
            beam_score = torch.tensor([b.acc_score for b in newest_branch], device=model.device)
            beam_score = beam_score.view((1, 1, beam_width, 1))
            token_scores = log_probs + beam_score
            vocab_size = token_scores.shape[-1]
            token_scores = token_scores.view(beam_width * vocab_size)
            topk_scores, tokens = torch.topk(
                token_scores, beam_width * max(2, 1 + len(eos_token_id)),
                dim=0, largest=True, sorted=True
            )
            next_indices = torch.div(tokens, vocab_size, rounding_mode="floor")
            tokens = tokens % vocab_size
            tmp_newest_branch = []
            step_tokens = []
            step_parents = []
            for j in range(len(tokens)):
                token_id = tokens[j]
                node = SearchNode(searchTree, idx, token_id, topk_scores[j])
                if token_id in eos_token_id:
                    node.parent = newest_branch[next_indices[j]]
                    node.idx = -1
                else:
                    newest_branch[next_indices[j]].add_children(node)
                    tmp_newest_branch.append(node)
                    idx += 1
                    step_tokens.append(token_id.item())
                    step_parents.append(int(next_indices[j]))
                if len(tmp_newest_branch) >= beam_width:
                    break
            newest_branch = tmp_newest_branch
            if step_tokens:
                tree_steps.append({"tokens": step_tokens, "parents": step_parents})
            if len(newest_branch) == 0:
                break
    return logits_list, tree_steps


def main():
    parser = argparse.ArgumentParser(description="Compare logits between trie decoding and baseline beam search")
    parser.add_argument("--model", choices=MODEL_CHOICES.keys(), help="Model choice")
    parser.add_argument("--dataset", choices=DATASET_CHOICES.keys(), help="Dataset choice")
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (omit to use all)"
    )
    parser.add_argument("--beam_width", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    set_seed(args.seed)

    model_name = MODEL_CHOICES[args.model]
    ds_info = DATASET_CHOICES[args.dataset]

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    dataset = load_dataset(ds_info["path"], ds_info["config"], split=ds_info["split"])
    if args.samples is not None:
        n = min(args.samples, len(dataset))
        dataset = dataset.select(range(n))

    # build fixed output path
    safe_model = args.model.replace("/", "_")
    output_path = os.path.join(
        "reproduction",
        "final_out",
        "logits",
        safe_model,
        f"{args.dataset}.json",
    )

    records = []
    eos = [tokenizer.eos_token_id]
    for sample in dataset:
        prompt = sample[ds_info["text_column"]]
        tree_logits, tree_steps = record_trie_logits(
            model,
            tokenizer,
            prompt,
            args.beam_width,
            args.max_new_tokens,
            eos,
        )
        base_logits, base_steps = record_baseline_logits(
            model,
            tokenizer,
            prompt,
            args.beam_width,
            args.max_new_tokens,
            eos,
        )
        records.append({
            "prompt": prompt,
            "tree": [t.tolist() for t in tree_logits],
            "baseline": [b.tolist() for b in base_logits],
            "tree_structure": tree_steps,
            "baseline_structure": base_steps,
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f)


if __name__ == "__main__":
    main()

    """
    Example usage:
    python -m reproduction.logit_test \
        --model llama3 \
        --dataset human_eval \
        --samples 10


    Available models: llama3, phi35, mistral, llama3_70b
    Available datasets: human_eval, gsm8k, cnn, wmt
    """
