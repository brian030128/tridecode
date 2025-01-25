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


## Notes
Code in reproduction is not ready for production environments, as the trie traversal implementation uses recursive functions, which can lead to stack overflow errors for long outputs.
A production library is under development and will be released soon.


