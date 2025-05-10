import json
from task import Gsm8kTask
import os

source = {
    "LLAMA3": {
        "origin": ["1_1000.jsonl", "3_1000.jsonl", "9_1000.jsonl", "15_1000.jsonl"],
        "tree": ["3_1000.jsonl", "9_1000.jsonl", "15_1000.jsonl"]
    },
    "PHI35": {
        "origin": ["1_1000.jsonl", "3_1000.jsonl", "9_1000.jsonl", "15_1000.jsonl"],
        "tree": ["3_1000.jsonl", "9_1000.jsonl", "15_1000.jsonl"]
    },
}

for model in source.keys():
    for decode_type in source[model].keys():
        for file_name in source[model][decode_type]:
            path = f"out/{model}/{decode_type}/GSM8K/{file_name}"
            

            prompt = []
            with open(path, 'r') as f:
                data = [json.loads(line) for line in f]
                ground = Gsm8kTask().get_ds().select(range(100))
                
                for idx, d in enumerate(data):
                    answer = ground[idx]['answer']
                    prompt.append(
                            {"idx": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [
                                {"role": "system", "content": """
You are a grading, system, you will recieve a ground truth answer and a student answer.
Output '1' if the student's answer is correct, output '0' otherwise. Only output 1 or 0.
"""},
                                {"role": "user", "content": f"""
##### Ground Truth #####
{answer}
##### Student Answer #####
{d['output']}
"""}],"max_tokens": 10}}
                    )
            batch_file = f"out/openai/{model}/{decode_type}/{file_name}"
            os.makedirs(f"out/openai/{model}/{decode_type}", exist_ok=True)
            with open(batch_file, 'w') as f:
                for p in prompt:
                    f.write(json.dumps(p) + '\n')





