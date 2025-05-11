import os
import json


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




                

def parse(in_folder, out_folder):
    all_ids = set()
    for i in range(164):
        all_ids.add(f"HumanEval/{i}")

    for filename in os.listdir(in_folder):
        file_path = os.path.join(in_folder, filename)
            
        # Check if it's a file (not a directory)
        if not os.path.isfile(file_path):
            continue
        print("reading", file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            with open(os.path.join(out_folder, filename), 'w') as out_file:
                all_id_copy = all_ids.copy()
                for line_number, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    try:
                        data = json.loads(line)

                        completion = data['output']
                        out_file.write(json.dumps({
                            "task_id": data['id'],
                            "completion": completion
                        }) + "\n")
                        all_id_copy.remove(data['id'])
                    except json.JSONDecodeError as e:
                        print(f"Error on line {line_number}: {e}")
                        print(f"Line content: {repr(line)}")
                for id in all_id_copy:
                    out_file.write(json.dumps({
                        "task_id": id,
                        "completion": ""
                    }) + "\n")
                print(f"missing {len(all_id_copy)} entries")


for model in source.keys():
    for decode_type in source[model].keys():
        for file_name in source[model][decode_type]:
            in_dir = f'out/{model}/{decode_type}/HUMAN_EVAL'
            out_dir = f'tmp_out/{model}/{decode_type}/HUMAN_EVAL'
            os.make_dirs(out_dir, exists_ok=True)
            parse(in_dir, out_dir)