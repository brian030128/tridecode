import json
from task import Gsm8kTask
import os

source = {
    # "LLAMA3": {
    #     "origin": ["1_1000.jsonl", "3_1000.jsonl", "9_1000.jsonl", "15_1000.jsonl"],
    #     "tree": ["3_1000.jsonl", "9_1000.jsonl", "15_1000.jsonl"]
    # },
    "PHI35": {
        "origin": ["1_1000.jsonl", "3_1000.jsonl", "9_1000.jsonl", "15_1000.jsonl"],
        "tree": ["3_1000.jsonl", "9_1000.jsonl", "15_1000.jsonl"]
    },
}

def create_batch_files():
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
                                {"custom_id": str(idx), "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [
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

from openai import OpenAI
client = OpenAI()
def send():
    for model in source.keys():
        for decode_type in source[model].keys():
            for file_name in source[model][decode_type]:
                path = f"out/openai/{model}/{decode_type}/{file_name}"
                batch_input_file = client.files.create(
                file=open(path, "rb"),
                    purpose="batch"
                )

                print(batch_input_file)

                batch_input_file_id = batch_input_file.id
                batch = client.batches.create(
                    input_file_id=batch_input_file_id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={
                        "description": path
                    }
                )

                while True:
                    batch = client.batches.retrieve(batch.id)
                    print(batch)
                    if batch.status == "completed":
                        break
                file_response = client.files.content(batch.output_file_id)

                scores = []
                for line in file_response.text.split('\n'):
                    if line.strip() == "":
                        break
                    obj = json.loads(line)
                    score = obj['response']['body']['choices'][0]['message']['content']
                    scores.append(score)

                in_path = f"out/{model}/{decode_type}/GSM8K/{file_name}"
                o_path = f"final_out/{model}/{decode_type}/GSM8K/{file_name}"
                os.makedirs(f'final_out/{model}/{decode_type}/GSM8K/', exist_ok=True)
                entries = []

                with open(in_path, 'r') as f:
                    for idx, line in enumerate(f):
                        obj = json.loads(line)
                        obj['score'] = scores[idx]
                        entries.append(obj)
                
                with open(o_path, 'w') as f:
                    for entry in entries:
                        f.write(json.dumps(entry) + '\n')


create_batch_files()

send()



