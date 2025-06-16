import glob
import re
import json

from tqdm import tqdm
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from huggingface_hub import login

def load_json(json_path):
    if json_path.endswith('.json'):
        with open(json_path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                return [json.loads(line) for line in f if line.strip()]
    elif json_path.endswith('.jsonl'):
        data = []
        with open(json_path, "r") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    else:
        return None

def parse_task_data(text_data: str) -> list:
    """
    Parses a string containing task data to extract task type, question, and answer.
    Handles multiple Q: A: pairs within a single text block.

    Args:
        text_data: The input string containing the task information.

    Returns:
        A list of dictionaries, where each dictionary represents a parsed task
        and contains 'task_type', 'question', and 'answer' keys.
    """
    parsed_results = []

    # Split the input into multiple <task:...> blocks
    task_blocks = re.findall(r'(<task:[^>]+>.*?)(?=<task:|$)', text_data, re.DOTALL)

    for block in task_blocks:
        # Extract task type
        task_type_match = re.search(r'<task:([^>]+)>', block)
        task_type = task_type_match.group(1) if task_type_match else "unknown"

        # Remove the task tag for easier processing
        clean_block = re.sub(r'<task:[^>]+>', '', block, 1).strip()

        # Match everything from start up to <PRED>A: as question, then capture answer
        qa_pairs = re.findall(r'(.*?)Q: (.*?) <PRED>A: (.*?)</PRED>', clean_block, re.DOTALL)

        for prefix, q_suffix, raw_answer in qa_pairs:
            # Combine both parts of the question
            question = (prefix + "Q: " + q_suffix).strip()

            # Clean answer by removing nested tags
            answer = re.sub(
                r'<PRED:ANSWER>|<PRED:DISCRETE>|<PRED:BINARY>|</PRED:BINARY>|</PRED:DISCRETE>|</PRED:ANSWER>|\n',
                '',
                raw_answer
            ).strip()

            parsed_results.append({
                "question_type": task_type,
                "question": question,
                "answer": answer
            })

    return parsed_results

def preprocess(dataset_path):
    dataset = []
    for file_path in glob.glob(dataset_path):
        data = load_json(file_path)
        if isinstance(data, dict):
            dataset.append(data)
        else:
            dataset.extend(data)
    
    all_parsed_instances = []
    for raw_json_data in tqdm(dataset):
        text_content = raw_json_data['text']
        parsed_tasks = parse_task_data(text_content)
        original_uid = raw_json_data.get('uid')
        original_video = raw_json_data.get('video')

        for task_instance in parsed_tasks:
                # Add original UID and video to each new instance
                instance_with_metadata = {
                    "uid": original_uid, # Or generate a new unique ID if needed for sub-tasks
                    "video": original_video,
                    **task_instance # Unpack the task_type, question, answer
                }
                all_parsed_instances.append(instance_with_metadata)
    
    return Dataset.from_list(all_parsed_instances)

if __name__=='__main__':
    dataset_dict = {}
    split_list = ['train', 'val']
    for split in split_list:
        dataset_path = f"/data/klx/hf_dataset/robovqa/gdm-robovqa/json/{split}/*.json"
        dataset_dict[split] = preprocess(dataset_path)
        print(f"Finished processing {split} split. Number of instances: {len(dataset_dict[split])}")
    
    dataset_dict = DatasetDict(dataset_dict)
    
    # Define the path where you want to save your dataset
    # login()  
    dataset_dict.push_to_hub("koulx/RoboVQA_TF2HF")
    print("Dataset saved successfully!")