from model_type import ModelType

from enum import Enum

import datasets
from datasets import load_dataset

class TaskType(Enum):
    HUMAN_EVAL = 1
    GSM8K = 2
    CNN = 3 


class Task:
    def phi(self, prompt) -> str:
        pass
    def mistral(self, prompt) -> str:
        pass
    def llama3(self, prompt) -> str:
        pass
    def get_prompt(self, model: ModelType, prompt: str) -> str:
        match model:
            case ModelType.PHI35:
                return self.phi(prompt)
            case ModelType.LLAMA3:
                return self.llama3(prompt)
            case ModelType.MISTRAL:
                return self.mistral(prompt)
            
    def extract_answer(self, text) -> str:
        pass

    def type(self) -> TaskType:
        pass

    def get_ds():
        pass

def get_task(type: TaskType) -> Task:
    match type:
        case TaskType.HUMAN_EVAL:
            return HumanEvalTask()
        case TaskType.GSM8K:
            return Gsm8kTask()
        case TaskType.CNN:
            return CNNSumTask()
        
    

class HumanEvalTask(Task):
    
    def llama3(self, prompt) -> str:
        return f"""<|start_header_id|>system<|end_header_id|>
                You are a programmer.
                <|eot_id|><|start_header_id|>user<|end_header_id|>
Complete the following function. No explaination is needed, output the code directly.
{prompt}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                """
    def mistral(self, prompt) -> str:
        return f"""
                <s>[SYSTEM_PROMPT]You are a programmer.[/SYSTEM_PROMPT][INST]Complete the following function. No explaination is needed, output the code directly.
                {prompt}[/INST]
            """
    
    def phi(self, prompt) -> str:
        return f"""<|system|>
You are a programmer.<|end|>
<|user|>
Complete the following function. No explaination is needed, output the code directly.
{prompt}<|end|>
<|assistant|>"""

    
    def extract_answer(self, text) -> str:
        function_lines = []
        in_function = False
        base_indent = None
        
        lines = text.split('\n')

        for line in lines:
            # Detect function start
            if 'def ' in line and not in_function:
                in_function = True
                base_indent = len(line) - len(line.lstrip())
                function_lines.append(line[line.find("def"):])
                continue

            # If we're inside a function
            if in_function:
                # If empty line, add it
                if not line.strip():
                    function_lines.append(line)
                    continue

                # Check line indentation
                current_indent = len(line) - len(line.lstrip())
                
                # If indentation is less than or equal to base, we're out of the function
                if current_indent <= base_indent and line.strip() and not line.strip().startswith('#'):
                    break

                # Add the line to our function with original indentation
                function_lines.append(line)


        return '\n'.join(function_lines)
    
    def type(self) -> TaskType:
        return TaskType.HUMAN_EVAL


    def get_ds(self) -> datasets.Dataset:
        ds = load_dataset("openai_humaneval", split='test')
        ds = ds.map(
            HumanEvalTask.convert_human_eval_format,
            batched=True
        )
        return ds

    @staticmethod
    def convert_human_eval_format(d):
        return {
            'id': d['task_id'],
            'text': d['prompt']
        }


class Gsm8kTask(Task):

    def llama3(self, prompt) -> str:
        return f"""<|start_header_id|>system<|end_header_id|>
                Solve the math problem.
                <|eot_id|><|start_header_id|>user<|end_header_id|>
{prompt}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                """
    def mistral(self, prompt) -> str:
        return f"""
                <s>[SYSTEM_PROMPT]Solve the math problem.[/SYSTEM_PROMPT][INST]
                {prompt}[/INST]
            """
    
    def phi(self, prompt) -> str:
        return f"""<|system|>
Solve the math problem.<|end|>
<|user|>
{prompt}<|end|>
<|assistant|>"""

    def extract_answer(self, text) -> str:
        return text
    
    def type(self) -> TaskType:
        return TaskType.GSM8K


    def get_ds(self) -> datasets.Dataset:
        ds = load_dataset("openai/gsm8k", 'main', split='test')
        ds = ds.map(
            Gsm8kTask.convert_format,
            batched=True
        )
        return ds

    @staticmethod
    def convert_format(d):
        return {
            'text': d['question'],
            'answer': d['answer']
        }


class Gsm8kTask(Task):

    def llama3(self, prompt) -> str:
        return f"""<|start_header_id|>system<|end_header_id|>
                Solve the math problem.
                <|eot_id|><|start_header_id|>user<|end_header_id|>
{prompt}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                """
    def mistral(self, prompt) -> str:
        return f"""
                <s>[SYSTEM_PROMPT]Solve the math problem.[/SYSTEM_PROMPT][INST]
                {prompt}[/INST]
            """
    
    def phi(self, prompt) -> str:
        return f"""<|system|>
Solve the math problem.<|end|>
<|user|>
{prompt}<|end|>
<|assistant|>"""

    def extract_answer(self, text) -> str:
        return text
    
    def type(self) -> TaskType:
        return TaskType.GSM8K


    def get_ds(self) -> datasets.Dataset:
        ds = load_dataset("openai/gsm8k", 'main', split='test')
        ds = ds.map(
            Gsm8kTask.convert_format,
            batched=True
        )
        return ds

    @staticmethod
    def convert_format(d):
        return {
            'text': d['question'],
            'answer': d['answer']
        }



class CNNSumTask:
    def llama3(self, prompt) -> str:
        return f"""<|start_header_id|>system<|end_header_id|>
                Output the highlight of the news.
                <|eot_id|><|start_header_id|>user<|end_header_id|>
{prompt}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                """
    def mistral(self, prompt) -> str:
        return f"""
                <s>[SYSTEM_PROMPT]Output the highlight of the news.[/SYSTEM_PROMPT][INST]
                {prompt}[/INST]
            """
    
    def phi(self, prompt) -> str:
        return f"""<|system|>
Output the highlight of the news.<|end|>
<|user|>
{prompt}<|end|>
<|assistant|>"""
    def get_prompt(self, model: ModelType, prompt: str) -> str:
        match model:
            case ModelType.PHI35:
                return self.phi(prompt)
            case ModelType.LLAMA3:
                return self.llama3(prompt)
            case ModelType.MISTRAL:
                return self.mistral(prompt)
            
    def extract_answer(self, text) -> str:
        return text

    def type(self) -> TaskType:
        return TaskType.CNN

    @staticmethod
    def convert_format(d):
        return {
            'id': d['id'],
            'text': d['article'],
            'answer': d['highlights']
        }

    def get_ds():
        ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split='train+validation+test')
        ds = ds.map(
            CNNSumTask.convert_format,
            batched=True,
            remove_columns=['article', 'highlights']
        )
        return ds