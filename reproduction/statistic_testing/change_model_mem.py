import os
import json

# Mapping from model_name to correct model_memory
MODEL_MEMORY = {
    'LLAMA3': 32121.061888,
    'PHI35': 15284.596736000001,
    'MISTRAL': 94289.63328
}

# Root directory containing your data
ROOT_DIR = 'reproduction/final_out'

for dirpath, _, filenames in os.walk(ROOT_DIR):
    # only process .jsonl files
    for fname in filenames:
        if not fname.endswith('.jsonl'):
            continue

        filepath = os.path.join(dirpath, fname)
        # derive model_name from the path: e.g. final_out/LLAMA3/origin/...
        rel_parts = os.path.relpath(dirpath, ROOT_DIR).split(os.sep)
        model_name = rel_parts[0]  # first component

        if model_name not in MODEL_MEMORY:
            print(f"⚠️  Skipping {filepath}, unknown model '{model_name}'")
            continue

        new_memory = MODEL_MEMORY[model_name]

        # read & update all lines
        updated = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                obj['model_memory'] = new_memory
                updated.append(json.dumps(obj, ensure_ascii=False))

        # write back
        with open(filepath, 'w', encoding='utf-8') as f:
            for line in updated:
                f.write(line + '\n')

        print(f"✅  Updated model_memory in {filepath}")

"""
Example usage:
python -m reproduction.statistic_testing.change_model_mem
"""