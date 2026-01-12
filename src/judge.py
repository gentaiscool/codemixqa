import json
import os
import argparse
import logging
from functools import partial
import concurrent.futures
import time
import tempfile
import re

from datasets import load_dataset
from tqdm import tqdm

from .constants import *
from .utils import *

# Global model variables
from transformers import AutoTokenizer
MODEL = None
TOKENIZER = None
OPENAI_CLIENT = None
GOOGLE_CLIENT = None

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# Grading template from evaluate.py
SYSTEM_GRADER_TEMPLATE = """
# Instruction

Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.

The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k".
    - Predicted answers "120k", "124k", and 115k" are all CORRECT.
    - Predicted answers "100k" and "113k" are INCORRECT.
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name.
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".

Now you will be given a new question, gold target, and predicted answer. Return exactly one letter:
A = CORRECT
B = INCORRECT
C = NOT_ATTEMPTED

Output ONLY the letter A, B, or C. Do not output anything else.
""".strip()

USER_GRADER_TEMPLATE = """
# Question

{question}

# Gold Target

{target}

# Predicted Answer

{predicted_answer}

# Your Grade (ONLY A, B, or C)
"""

CHOICE_LETTERS = ["A", "B", "C"]
CHOICE_STRINGS = ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
CHOICE_LETTER_TO_STRING = dict(zip(CHOICE_LETTERS, CHOICE_STRINGS))
DEFAULT_GRADE_IF_UNPARSEABLE = "C"  # Corresponds to NOT_ATTEMPTED

def format_grader_prompt(question: str, target: str, predicted_answer: str) -> str:
    return USER_GRADER_TEMPLATE.format(
        question=question,
        target=target,
        predicted_answer=predicted_answer,
    )

def initialize_model(config):
    """Initialize the appropriate model based on config"""
    global MODEL, TOKENIZER, OPENAI_CLIENT, GOOGLE_CLIENT
    from transformers import AutoTokenizer

    model_type = config.get('model_type', 'vllm')
    model_name = config.get('model_name')

    if model_type == 'vllm':
        if MODEL is None:
            from vllm import LLM
            MODEL = LLM(model=model_name, **config.get("model_args", {}))
        if TOKENIZER is None:
            TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    elif model_type == 'openai':
        if OPENAI_CLIENT is None:
            from openai import OpenAI
            api_key = config.get('api_key', os.environ.get('OPENAI_API_KEY'))
            OPENAI_CLIENT = OpenAI(api_key=api_key, timeout=config.get('timeout', 1500))
    elif model_type == 'google':
        if GOOGLE_CLIENT is None:
            from google import genai
            api_key = config.get('api_key', os.environ.get('GOOGLE_API_KEY'))
            GOOGLE_CLIENT = genai.Client(api_key=api_key)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

def generate_vllm(prompts, config):
    """Generate using vLLM"""
    from vllm import SamplingParams
    global MODEL, TOKENIZER

    sampling_params = SamplingParams(**config.get("generation_args", {}))

    formatted_prompts = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_GRADER_TEMPLATE},
            {"role": "user", "content": prompt},
        ]

        formatted_prompt = TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted_prompts.append(formatted_prompt)

    outputs = MODEL.generate(formatted_prompts, sampling_params)

    return [output.outputs[0].text for output in outputs]

def generate_openai(prompt, config):
    """Generate using OpenAI API"""
    global OPENAI_CLIENT

    model_name = config.get('model_name')
    reasoning_effort = config.get('reasoning_effort', None)

    if model_name in ["gpt-5.2", "gpt-5"]:
        completion = OPENAI_CLIENT.chat.completions.parse(
            model=model_name,
            reasoning_effort=reasoning_effort,
            messages=[{"role": "system", "content": SYSTEM_GRADER_TEMPLATE}, {"role": "user", "content": prompt}],
            service_tier="flex",
            timeout=config.get('timeout', 1500)
        )
    else:
        completion = OPENAI_CLIENT.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": SYSTEM_GRADER_TEMPLATE}, {"role": "user", "content": prompt}],
        )
    return completion.choices[0].message.content

def generate_google(prompt, config):
    """Generate using Google Gemini API"""
    global GOOGLE_CLIENT
    from google.genai import types

    model_name = config.get('model_name')
    thinking_level = config.get('thinking_level', 'low')

    response = GOOGLE_CLIENT.models.generate_content(
        model=model_name,
        contents=f"{SYSTEM_GRADER_TEMPLATE}\n\n{prompt}",
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level)
        )
    )
    return response.text

def parse_grade_letter(grading_response_text):
    """Parse grade letter from LLM response"""
    match = re.search(r"(A|B|C)", grading_response_text)
    if match:
        return match.group(0)
    else:
        if "CORRECT" in grading_response_text.upper(): return "A"
        if "INCORRECT" in grading_response_text.upper(): return "B"
        if "NOT_ATTEMPTED" in grading_response_text.upper(): return "C"
        logging.warning(f"Could not parse grade from: '{grading_response_text}'. Defaulting to {DEFAULT_GRADE_IF_UNPARSEABLE}.")
        return DEFAULT_GRADE_IF_UNPARSEABLE

def client_completion(config, final_dataset):
    """Run inference on the judge model"""
    global MODEL, TOKENIZER

    initialize_model(config)

    model_type = config.get('model_type', 'vllm')

    if model_type == 'vllm':
        # Batch processing for vLLM
        batched_prompts = [input_item['prompt'] for input_item in final_dataset]
        generated_texts = generate_vllm(batched_prompts, config)

        results = []
        for input_item, generated_text in zip(final_dataset, generated_texts):
            grade_letter = parse_grade_letter(generated_text)
            results.append({
                "id": input_item['id'],
                "question": input_item['question'],
                "gold_target": input_item['gold_target'],
                "predicted_answer": input_item['predicted_answer'],
                "grade_letter": grade_letter,
                "grade_str": CHOICE_LETTER_TO_STRING.get(grade_letter, "UNKNOWN"),
                "is_correct": grade_letter == "A",
                "is_incorrect": grade_letter == "B",
                "is_not_attempted": grade_letter == "C",
                "grader_response": generated_text
            })
    else:
        # Sequential processing for OpenAI/Google
        results = []
        for input_item in tqdm(final_dataset, desc="Grading"):
            prompt = input_item['prompt']

            if model_type == 'openai':
                generated_text = generate_openai(prompt, config)
            elif model_type == 'google':
                generated_text = generate_google(prompt, config)

            grade_letter = parse_grade_letter(generated_text)
            results.append({
                "id": input_item['id'],
                "question": input_item['question'],
                "gold_target": input_item['gold_target'],
                "predicted_answer": input_item['predicted_answer'],
                "grade_letter": grade_letter,
                "grade_str": CHOICE_LETTER_TO_STRING.get(grade_letter, "UNKNOWN"),
                "is_correct": grade_letter == "A",
                "is_incorrect": grade_letter == "B",
                "is_not_attempted": grade_letter == "C",
                "grader_response": generated_text
            })

    return results

def process_dataset_in_chunks(dataset_path, dataset_type, output_path, chunk_size, start_offset, end_offset, model_config,
                              safe_infer=False, debug=False):
    """Process the entire dataset in chunks"""
    total_size = 0
    if dataset_type in TRAIN_DATASETS_DICT_SIZE:
        total_size = TRAIN_DATASETS_DICT_SIZE[dataset_type]
        dataset = load_dataset(TRAIN_DATASETS_DICT[dataset_type], split='train')
    else:
        total_size = EVAL_DATASETS_DICT_SIZE[dataset_type]
        dataset = load_dataset(EVAL_DATASETS_DICT[dataset_type], split='test')

    if end_offset > 0:
        total_size = min(total_size, end_offset)

    # Load the response data (predictions to be graded)
    with open(dataset_path, 'r') as f:
        response_data = json.load(f)

    # Create a mapping of response data by ID for quick lookup
    response_map = {}
    for item in response_data:
        response_map[item['id']] = {
            "reasoning_content": item.get("reasoning_content"),
            "response": item.get("response"),
        }

    all_results = []

    for offset in range(start_offset, total_size, chunk_size):
        logging.info(f"Processing offset: {offset}")

        # Get the current chunk of dataset rows
        chunk_end = min(offset + chunk_size, total_size)
        chunk_dataset = []

        for idx in range(offset, chunk_end):
            row = dataset[idx]
            row_id = row.get('id', idx)  # Use row's ID or index as fallback

            # Check if we have a response for this row
            if row_id not in response_map:
                # logging.warning(f"No response found for row ID: {row_id}")
                continue

            candidate_answer = response_map[row_id].get("response")

            # Prepare input text and ground truth based on dataset type
            question = row.get('problem')
            gold_target = row.get('answer')

            # Create the judge prompt using the grading template
            grader_prompt = format_grader_prompt(question, gold_target, candidate_answer)

            chunk_dataset.append({
                "id": row_id,
                "prompt": grader_prompt,
                "question": question,
                "gold_target": gold_target,
                "predicted_answer": candidate_answer
            })

        if chunk_dataset is not None and len(chunk_dataset) > 0:
            logging.info(f"Processing {len(chunk_dataset)} samples")

            # Generate responses for this chunk
            results = client_completion(model_config, chunk_dataset)
            all_results.extend(results)

            # Write intermediate results
            write_results(results, output_path)

            # If debug mode, only process one chunk
            if debug:
                break

    return all_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on any evaluation datasets')
    parser.add_argument('--response_path', '-r', type=str, required=True,
                        help="Response path to load the data.")
    parser.add_argument('--dataset_names', '-d', type=str, required=True,
                        help="Dataset name to load the data.")
    parser.add_argument('--model_config_path', '-c', type=str, required=True,
                        help=f"Jugde model's config for running evaluation. For example, see `configs`.")
    parser.add_argument('--output_save_path', '-o', type=str, default='output',
                        help="Output folder name.")
    parser.add_argument('--chunk_size', type=int, default=-1,
                        help="Save batch size.")
    parser.add_argument('--start_offset', type=int, default=0,
                        help="Start offset.")
    parser.add_argument('--end_offset', type=int, default=-1,
                        help="End offset.")
    parser.add_argument(
        '--seeds_list',
        type=int,
        nargs='+',
        default=[0, 1, 2], # Changed default to a list, as it will now store a list
        help="List of seeds to use. Provide one or more integers separated by spaces (e.g., --seeds_list 0 1 2). Defaults to [0, 1, 2]."
    )
    parser.add_argument('--safe-infer', action="store_true", dest="safe_infer",
                        help=f"Filter out input that is longer than max-model-len minus output length.")
    parser.add_argument("--debug", action="store_true", dest="debug",
                        help=f"Debug with {DEBUG_COUNT} samples.")
    parser.set_defaults(safe_infer=False, surgery=False, debug=False)
    args = parser.parse_args()
    
    logging.info("==== Current Arguments ====")
    logging.info(args)
    logging.info("=== End of Current Arguments ====")

    config_path = args.model_config_path.strip()
    model_config = {}
    if not os.path.exists(config_path):
        config_abs_path = os.path.join(ROOT_DIR, config_path)
        if not os.path.exists(config_path):
            config_abs_path = os.path.join(ROOT_DIR, config_path)
            if not os.path.exists(config_abs_path):
                raise ValueError(f"Config path `{config_path}` does not exist!")
            else:
                logging.warning(f"Config path `{config_path}` is not found, switching to `{config_abs_path}`")
                config_path = config_abs_path
    else:
        config_abs_path = config_path
    
    if not config_abs_path.endswith('.json'):
        raise NotImplementedError("Config path is not in JSON Format, other format is not implemented yet!")
    else:
        with open(config_abs_path, 'r') as f:
            model_config = json.load(f)

    # Create output folder
    output_save_dir = args.output_save_path
    if not os.path.isabs(output_save_dir):
        output_save_dir = os.path.join(ROOT_DIR, output_save_dir)
        
    dataset_names = args.dataset_names.strip()
    eval_dataset_list = []
    if dataset_names == "all":
        eval_dataset_list = list(EVAL_DATASETS_DICT.keys())
    else:
        dataset_name_list = dataset_names.split(",")
        for dataset_name in dataset_name_list:
            if dataset_name in TRAIN_DATASETS_DICT.keys() or dataset_name in EVAL_DATASETS_DICT.keys():
                eval_dataset_list.append(dataset_name)
            else:
                logging.warning(f"Unrecognized evaluation dataset named `{dataset_name}`, skipping ...")

    if len(eval_dataset_list) == 0:
        raise ValueError("Evaluation datasets cannot be empty!")
        
    os.makedirs(output_save_dir, exist_ok=True)

    if 'generation_args' not in model_config:
        model_config['generation_args'] = {}

    for seed in args.seeds_list:
        for dataset_name in eval_dataset_list:
            dataset_path = os.path.join(args.response_path, f"{dataset_name}_{seed}.json")
            output_save_path = os.path.join(output_save_dir, f"{dataset_name}_{seed}_graded.json")

            logging.info(f"Processing dataset: {dataset_name}, seed: {seed}")

            if args.chunk_size > 0:
                process_dataset_in_chunks(
                    dataset_path=dataset_path,
                    dataset_type=dataset_name,
                    output_path=output_save_path,
                    chunk_size=args.chunk_size,
                    start_offset=args.start_offset,
                    end_offset=args.end_offset,
                    model_config=model_config,
                    safe_infer=args.safe_infer,
                    debug=args.debug
                )
            else:
                process_dataset_in_chunks(
                    dataset_path=dataset_path,
                    dataset_type=dataset_name,
                    output_path=output_save_path,
                    chunk_size=1000000,  # Large number to get all data at once
                    start_offset=args.start_offset,
                    end_offset=args.end_offset,
                    model_config=model_config,
                    safe_infer=args.safe_infer,
                    debug=args.debug
                )

            logging.info(f"Graded results saved to: {output_save_path}")

    logging.info("\n" + "="*80)
    logging.info("Grading completed! Run analyze_judge.py to view statistics.")
    logging.info("="*80)