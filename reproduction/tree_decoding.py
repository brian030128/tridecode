import torch
import gc as gpu_gc
from transformers import LlamaForCausalLM
from typing import Tuple, List, Optional
from torch import Tensor
from transformers.cache_utils import DynamicCache

from typing import List, Tuple
import time

from typing import List, Tuple
import GPUtil

import torch
import gc as gpu_gc


def get_gpu_usage():
    gpus = GPUtil.getGPUs()
    return gpus[0].memoryUsed
    
minFloat = torch.finfo(torch.float).min
device = "cuda" if torch.cuda.is_available() else "cpu"
class SearchNode:
    def __init__(self, root, idx, token_id, token_score):
        self.root: 'SearchTree' = root
        self.idx: int = idx
        self.token_id: Tensor = token_id
        self.token_score: torch.FloatTensor = token_score
        self.parent: Optional['SearchNode'] = None
        self.children: List['SearchNode'] = []
        self.acc_score: torch.FloatTensor = token_score


    def add_children(self, child):
        self.children.append(child)
        child.parent = self
        self.root.node_count += 1

    def delete_child(self, child):
        self.children.remove(child)
        self.root.node_count -= 1


class SearchTree:
    def __init__(self,model, beam_width=3):
        self.node_count: int = 0
        self.model = model
        self.device = model.device
        self.root: List[SearchNode] = []
        self.beam_width: int = beam_width


def cleanup_node(node: SearchNode):
    node.token_id = None
    node.token_score = None
    node.acc_score = None

def dfs(searchNode: SearchNode, targets: List[int], traversed: List[int]) -> Tuple[bool, List[int], List[int]]:
    # returns found, found path, unused nodes
    traversed.append(searchNode.idx)
    if searchNode.idx in targets:
        return (True, traversed, [])
    
    if len(searchNode.children) == 0:
        return (False, [], traversed)
    
    if len(searchNode.children) == 1:
        return dfs(searchNode.children[0], targets, traversed)
    
    child_found = False
    found_path = []
    unused = []
    for child in searchNode.children:
        found, fp, u = dfs(child, targets, [])
        if found:
            found_path += fp
            child_found = True
        unused += u

    if child_found:
        found_path = traversed + found_path
    else:
        unused = traversed + unused
    
    return (child_found, found_path, unused)


def determine_unused_nodes(searchTree: SearchTree, targets: List[int]) -> Tuple[List[int], List[int]]:
    all_unused = []
    all_used = []
    for child in searchTree.root:
        _, used, unused = dfs(child, targets, [])
        all_unused += unused
        all_used += used
    return (all_used, all_unused)


def generate_causal_mask(searchTree: SearchTree,input_len: int,nodes: List[SearchNode]) -> torch.Tensor:
    branch_count = len(nodes)
    mask = torch.full((1, 1, branch_count, searchTree.node_count + input_len), minFloat, device=device, dtype=torch.float)
    mask[0, 0,:,:input_len] = 0
    tmp = nodes.copy()
    #print("========")
    while True:
        end = False
        for i in range(branch_count):
            #print(i, tmp[i].idx)
            mask[0, 0, i, tmp[i].idx + input_len] = 0
            if tmp[i].parent is not None:
                tmp[i] = tmp[i].parent
            else:
                end = True
        if end:
            return mask


def print_tree_state(searchTree: SearchTree,nodes: List[SearchNode]):
    branch_count = len(nodes)
    tmp = nodes.copy()
    print("========")
    print("node count: ", searchTree.node_count)
    while True:
        end = False
        for i in range(branch_count):
            print(i, tmp[i].idx)
            if tmp[i].parent is not None:
                tmp[i] = tmp[i].parent
            else:
                end = True
        if end:
            return

import torch
import torch.nn.functional as F
from collections import deque
from transformers.models import metrics


def prune_kv_cache(past_key_values, input_length, remove_idx: List[int]):
    device = past_key_values[0][0].device
    remove_idx = [i + input_length for i in remove_idx]
    #print("remove", remove_idx)
    all_indices = torch.arange(past_key_values[0][0].size(2), device = device)

    keep_indices = all_indices[~torch.isin(all_indices, torch.tensor(remove_idx, device=device))]
    #print("keep", keep_indices)

    for i in range(len(past_key_values)):
        if keep_indices.device != past_key_values.key_cache[i].device:
            keep_indices= keep_indices.to(past_key_values.key_cache[i].device)
        past_key_values.key_cache[i] = torch.index_select(past_key_values.key_cache[i], 2, keep_indices)
        past_key_values.value_cache[i] = torch.index_select(past_key_values.value_cache[i], 2, keep_indices)

def clear_cache():
    torch.cuda.empty_cache()
    gpu_gc.collect()

def prune_tree(searchTree: SearchTree, remove_idx: List[int]):
    for child in searchTree.root[:]:
        if child.idx in remove_idx:
            #print("removed ", child.idx)
            searchTree.root.remove(child)
    tmp = deque(searchTree.root)
    while len(tmp) > 0:
        node = tmp.popleft()
        for child in node.children[:]:
            if child.idx in remove_idx:
                #print("removed ", child.idx)
                node.children.remove(child)
                tmp.append(child)
            else:
                tmp.append(child)

    i = 0

    tmp = deque(searchTree.root)
    while len(tmp) > 0:
        children = []
        while len(tmp) > 0:
            node = tmp.popleft()
            node.idx = i
            i += 1
            for child in node.children:
                children.append(child)
        children = sorted(children, key=lambda node: node.idx)
        tmp.extend(children)
    searchTree.node_count = i

def gc(searchTree: SearchTree,input_length, newest_branch: List[SearchNode], past_key_values):
    ignored = newest_branch
    unused = determine_unused_nodes(searchTree, [ node.idx for node in ignored])
    #print("Unused: ", len(unused[1]), len(unused[0]) + len(unused[1]) , unused)
    prune_tree(searchTree, unused[1])
    kv = prune_kv_cache(past_key_values,input_length, unused[1])
    #print_tree_state(searchTree, newest_branch)
    return 

import torch
import gc as gpu_gc

@torch.no_grad()
def generate_next_tokens(model, input_ids, beam_width = 3, max_new_tokens=300,eos_token_id: List[int] = [32000]) -> Tuple[torch.Tensor, List[int]]:
    early_complete = False
    gpu_usage = []
    device = model.device
    past_key_values = None
    input_len = input_ids.shape[1]
    print("input length: ", input_len)

    #generate the first 3 tokens
    past_key_values = DynamicCache()
    outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
    past_key_values = outputs.past_key_values
    token_scores = F.log_softmax(outputs.logits, dim=-1)

    token_scores, tokens = torch.topk(token_scores, beam_width, dim=-1, largest=True, sorted=True)
    searchTree = SearchTree(model,beam_width = beam_width)
    newest_branch: List[SearchNode] = []
    idx = 0

    n_eos_tokens = len(eos_token_id)
    n_tokens_to_keep = max(2, 1 + n_eos_tokens) * beam_width
    
    for i in range(beam_width):
        searchNode = SearchNode(searchTree, idx, tokens[0][-1][i], token_scores[0][-1][i])
        idx += 1
        newest_branch.append(searchNode)
        searchTree.root.append(searchNode)
        searchTree.node_count += 1
    
    completed_branches = []

    need_gc = False
    for i in range(input_len, max_new_tokens+ input_len):
        if  ((i % 15 == 0) or need_gc) and True:
           # print("gcccc")
            need_gc = False
            gc(searchTree,input_len, newest_branch, past_key_values)
            idx = searchTree.node_count
        #print("gpu: ", get_gpu_usage())
        position_ids = torch.tensor([[i for _ in range(beam_width)]], device=device)
        
        #construct attention_mask
        attention_mask = generate_causal_mask(searchTree,input_len , newest_branch)
        #print("attn", attention_mask.shape)
        #print("attn", attention_mask)
        #print(attention_mask[0][0])

        #construct input_ids
        input_ids = torch.tensor([[node.token_id for node in newest_branch]], device=device)
        
        #generate candidate tokens
        outputs = model(input_ids, past_key_values=past_key_values, position_ids=position_ids, attention_mask=attention_mask, use_cache=True)
        past_key_values = outputs.past_key_values
        #calculate token scores
        token_scores = F.log_softmax(outputs.logits, dim=-1)

        beam_score = torch.tensor([b.acc_score for b in newest_branch], device=model.device)
        beam_score = beam_score.view((1, 1, beam_width, 1))
        token_scores = token_scores + beam_score
        token_scores = token_scores.clone()

        vocab_size = token_scores.shape[-1]
        token_scores = token_scores.view(beam_width * vocab_size)
        token_scores, tokens = torch.topk(
            token_scores, n_tokens_to_keep, dim=0, largest=True, sorted=True
        )
        #which parent
        next_indices = torch.div(tokens, vocab_size, rounding_mode="floor") 
        #tokens
        tokens = tokens % vocab_size

        #update newest_branch and searchTree

        tmp_newest_branch = []
        
        completed_nodes = []
        picked = []
        picked_scores = []
        final_picked_parents = []
        
        for j in range(len(tokens)):
            token_id = tokens[j]
            picked.append(token_id.item())
            searchNode = SearchNode(searchTree, idx, token_id=token_id, token_score = token_scores[j])

            
            #print(int(token_idx/beam_width)," add child")
            
            if token_id in eos_token_id:
                #print(i, "ended")
                #need_gc = True
                completed_nodes.append(searchNode)
                completed_branches.append(searchNode)
                searchNode.parent = newest_branch[next_indices[j]]
                #tmp_newest_branch.append(searchNode)
                searchNode.idx = -1
            else:
                picked_scores.append(token_scores[j].item())
                newest_branch[next_indices[j]].add_children(searchNode)
                final_picked_parents.append(next_indices[j]) #- len(completed_nodes))
                idx += 1
                tmp_newest_branch.append(searchNode)

            if len(tmp_newest_branch) >= beam_width:
                break
        #print(i, picked_scores)
        next_indices = final_picked_parents
        newest_branch = tmp_newest_branch
        if early_complete:
            break
        if len(completed_branches) >= beam_width:
            early_complete = True
    
    #find the best branch
    max_score=0
    max_idx = 0
    for i in range(beam_width):
        if newest_branch[i].acc_score > max_score:
            max_score = newest_branch[i].acc_score
            max_idx = i

    #construct the output
    outputs = []
    if early_complete:
        newest_branch = completed_branches
    else:
        newest_branch = newest_branch + completed_branches
    for i in range(len(newest_branch)):
        output = torch.empty(0, device=model.device)
        branch_parent = newest_branch[i]
        length = 0
        score = branch_parent.acc_score
        while branch_parent is not None:
            length += 1
            output = torch.cat((output, branch_parent.token_id.unsqueeze(0)))
            branch_parent = branch_parent.parent
        output=output.flip(dims=[0])
        outputs.append((output, score / length))
        #outputs = torch.cat((outputs, output.unsqueeze(0)))
    max_score = max(x[1] for x in outputs)
    max_sequence = [x[0] for x in outputs if x[1] == max_score]
    return (max_sequence[0], metrics.memory_metrics, metrics.time_metrics)





def tree_warmup(model, tokenizer, prompt, num_beams, max_new_tokens, eos_token_id):
    tree_generate(model, tokenizer, prompt, num_beams, max_new_tokens, eos_token_id)

def tree_generate(model, tokenizer, prompt, num_beams, max_new_tokens, eos_token_id) -> Tuple[List[int], List[int], List[float]]:
    torch.cuda.empty_cache()
    gpu_gc.collect()
    metrics.clear()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output = generate_next_tokens(model, input_ids, beam_width=num_beams, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id)
    return (output[0].long(), output[1], output[2])


