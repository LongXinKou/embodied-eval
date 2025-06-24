'''
modified from "https://github.com/facebookresearch/open-eqa/blob/main/openeqa/evaluation/llm_match.py"

'''
import os
import numpy as np
import pandas as pd

from collections import defaultdict
from openai import OpenAI
from typing import Optional
from loguru import logger as eval_logger

client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY"), 
    base_url = os.getenv("OPENAI_API_BASE")
)

METRICS_FOR_OPENEQA_EMEQA = {
    "llm_match_score": "llm_match"
}
OPENEQA_EMEQA_QUESTION_TYPES = [
    "attribute recognition",
    "functional reasoning",
    "object localization",
    "object recognition",
    "object state recognition",
    "spatial understanding",
    "world knowledge",
]

def openeqa_emeqa_doc_to_visual(doc, dataset_kwargs=None):
    video_path = os.path.join(dataset_kwargs["video_dir"], f'{doc["episode_history"]}.mp4')
    if os.path.exists(video_path):
        video_path = video_path
    else:
        raise FileExistsError(f"video path:{video_path} does not exist.")
    return [video_path]

def openeqa_emeqa_doc_to_text(doc, dataset_kwargs=None):
    question = doc["question"]
    return question

def openeqa_emeqa_process_results(doc, results, dataset_kwargs=None):
    doc["prediction"] = results[0]
    pred_raw = results[0]
    
    target = doc["answer"]
    extra_answers = doc.get("extra_answers", None)
    question_type = doc["category"]

    result_dict = {"target": target}
    if extra_answers:
        result_dict = {"extra_answers": extra_answers}
    result_dict["question_type"] = question_type

    for key, value in METRICS_FOR_OPENEQA_EMEQA.items():
        score = eval(value)(doc["question"], target, pred_raw)
        doc[key] = {key: score}
        result_dict[key] = doc[key]

    return result_dict

def openeqa_emeqa_aggregate_results(results):
    for r in results:
        assert "question_type" in r, r
    results = pd.DataFrame(results)

    output = {}
    # key: {question_type}_{metric_name}
    for question_type, question_type_indexes in results.groupby("question_type").groups.items():
        per_question_type = results.iloc[question_type_indexes]
        if question_type in OPENEQA_EMEQA_QUESTION_TYPES:
            for metric in METRICS_FOR_OPENEQA_EMEQA.keys():
                metric_data = per_question_type[metric].tolist()
                avg_score = np.mean([x[metric] for x in metric_data])
                output[f"{question_type}_{metric}"] = avg_score
    
    metric_to_values = defaultdict(list)
    for key, val in output.items():
        if "_" in key:
            qtype, metric_name = key.rsplit("_", 1)
            if isinstance(val, (float, int)):
                metric_to_values[metric_name].append(val)
    for metric_name, vals in metric_to_values.items():
        if len(vals) > 0:
            avg_val = sum(vals) / len(vals)
            output[f"{metric_name}_average"] = avg_val

    output["overall"] = sum([_ for _ in output.values()]) / len(output)
    eval_logger.info(f"Evaluation results: {output}")
    return output

def llm_match(
        question: str,
        answer: str,
        prediction: str,
        extra_answers: Optional[list] = None,
        openai_model: str = "gpt-4o",
        openai_seed: int = 1234,
        openai_max_tokens: int = 32,
        openai_temperature: float = 0.2,
        verbose: bool = False,
    ):
    if prediction is None:
        return 0
    
    prompt_name = "mmbench" if extra_answers is None else "mmbench-extra"
    prompt = load_prompt(prompt_name)

    messages = prepare_openai_messages(
        prompt.format(
            question=question,
            answer=answer,
            prediction=prediction,
            extra_answers=extra_answers,
        ),
    )
    try:
        output = call_openai_api(
            messages=messages,
            model=openai_model,
            seed=openai_seed,
            max_tokens=openai_max_tokens,
            temperature=openai_temperature,
            verbose=verbose,
        )
        return parse_score(output)
    except Exception as e:
        eval_logger.info(f"Evaluation Error: {e}")

def prepare_openai_messages(content: str):
    return [{"role": "user", "content": content}]

def call_openai_api(
    messages: list,
    model: str = "gpt-4o",
    seed: Optional[int] = None,
    max_tokens: int = 32,
    temperature: float = 0.2,
    verbose: bool = False,
):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        seed=seed,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if verbose:
        print("openai api response: {}".format(completion))
    assert len(completion.choices) == 1
    return completion.choices[0].message.content

def parse_score(output: str, tag: str = "Your mark:") -> str:
    if output.isdigit():
        return int(output)
    start_idx = output.find(tag)
    if start_idx == -1:
        raise ValueError("Invalid output string: {}".format(output))
    end_idx = output.find("\n", start_idx)
    if end_idx == -1:
        return int(output[start_idx:].replace(tag, "").strip())
    return int(output[start_idx:end_idx].replace(tag, "").strip())

def load_prompt(name: str):
    if name == "mmbench":
        prompt = """
        You are an AI assistant who will help me to evaluate the response given the question and the correct answer.
        To mark a response, you should output a single integer between 1 and 5 (including 1, 5).
        5 means that the response perfectly matches the answer.
        1 means that the response is completely different from the answer.

        Example 1:
        Question: Is it overcast?
        Answer: no
        Response: yes
        Your mark: 1

        Example 2:
        Question: Who is standing at the table?
        Answer: woman
        Response: Jessica
        Your mark: 3

        Example 3:
        Question: Are there drapes to the right of the bed?
        Answer: yes
        Response: yes
        Your mark: 5

        Your Turn:
        Question: {question}
        Answer: {answer}
        Response: {prediction}    
        """
    else:
        prompt = """
        You are an AI assistant who will help me to evaluate the response given the question, the correct answer, and extra answers that are also correct.
        To mark a response, you should output a single integer between 1 and 5 (including 1, 5).
        5 means that the response perfectly matches the answer or any of the extra answers.
        1 means that the response is completely different from the answer and all of the extra answers.

        Example 1:
        Question: Is it overcast?
        Answer: no
        Extra Answers: ['doesn't look like it', 'no',' it's sunny']
        Response: yes
        Your mark: 1

        Example 2:
        Question: Who is standing at the table?
        Answer: woman
        Extra Answers: ['a woman', 'a lady', 'woman']
        Response: Jessica
        Your mark: 3

        Example 3:
        Question: Are there drapes to the right of the bed?
        Answer: yes
        Extra Answers: ['yes, there are drapes', 'yeah', 'the drapes are to the right of the king bed']
        Response: yes
        Your mark: 5

        Your Turn:
        Question: {question}
        Answer: {answer}
        Extra Answers: {extra_answers}
        Response: {prediction}
        """
    return prompt.strip()