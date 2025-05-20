# Paper Reproduction

## Presequites
Important!!!!!  
Please use a python virtual environment, as we modified the transformer library for evaluating resource consumption.  

Install requirements with  
```shell
pip install -r requirements.txt  
```

Run the experiments with `python main.py`.

```shell
python main.py
```

the results will be saved in the `out` folder.

Summarization tasks are scored with ROUGE-L automatically.
For HumanEval(code generation) , must be evaluated with https://github.com/openai/human-eval/tree/master .



