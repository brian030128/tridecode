import os
import json
import subprocess

import re

source = {
    "PHI35": {
        "origin": ["1_1000.jsonl", "3_1000.jsonl", "9_1000.jsonl", "15_1000.jsonl"],
        "tree": ["3_1000.jsonl", "9_1000.jsonl", "15_1000.jsonl"],
        "sample": ["sample.jsonl"]
    },
    "LLAMA3": {
        "origin": ["1_1000.jsonl", "3_1000.jsonl", "9_1000.jsonl", "15_1000.jsonl"],
        "tree": ["3_1000.jsonl", "9_1000.jsonl", "15_1000.jsonl"],
        "sample": ["sample.jsonl"]
    },
    "REASONING": {
        "origin": ["1_2000.jsonl", "3_2000.jsonl", "6_2000.jsonl"],
        "tree": ["3_1000.jsonl"],
        "sample": ["sample.jsonl"]
    }
}



def extract_box_content(text):
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None

                

def parse(in_folder, out_folder):

    for filename in os.listdir(in_folder):
        file_path = os.path.join(in_folder, filename)
            
        # Check if it's a file (not a directory)
        if not os.path.isfile(file_path):
            continue
        print("reading", file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            with open(os.path.join(out_folder, filename), 'w') as out_file:
                for line_number, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue

                    data = json.loads(line)
                    completion = data['output']
                    answer = data['answer']
                    #print(completion)
                    w = extract_box_content(completion)
                    print(w)
                    print(answer)
                    if w == answer:
                        print("match")
                        data["score"] = 1

                    out_file.write(json.dumps(data) + "\n")

import re



for model in source.keys():
    for decode_type in source[model].keys():
        for file_name in source[model][decode_type]:
            in_1_path = f"out/{model}/{decode_type}/MATH500"
            o_path = f"final_out/{model}/{decode_type}/MATH500/"
            os.makedirs(f'final_out/{model}/{decode_type}/MATH500/', exist_ok=True)
            parse(in_1_path, o_path)
