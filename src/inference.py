import json
import os
import argparse
import logging
from functools import partial
import concurrent.futures
import time
import tempfile
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm

from .constants import *
from .utils import *
from .dataset_gen import create_prompt_dataset

# Global model variables
MODEL = None

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

OPENAI_RETRIES = 3
MAX_TOKENS = 16384

def request_openai_completion(model_name, input_id, msg):
    OPENAI_CLIENT = OpenAI()
    for attempt in range(OPENAI_RETRIES):
        try:
            response = OPENAI_CLIENT.responses.create(
                model=model_name,
                input=msg[0]['content'], # Assuming single user message
                max_output_tokens=MAX_TOKENS,
                reasoning={ "effort": "medium"},
            )

            return {
                "id": input_id,
                "reasoning_content": None,
                "response": response.output[-1].content[0].text
            }
        except Exception as e:
            logging.warning(f"Error calling OpenAI API: {e}")
            time.sleep(61)
    logging.warning(f"Could not resolve error after {OPENAI_RETRIES} attempts for input ID: {input_id}")
    return {"id": input_id, "reasoning_content": None, "response": None}

def client_completion(config, final_dataset):
    from vllm import LLM, SamplingParams
    global MODEL

    batched_ids = [input_item['id'] for input_item in final_dataset]
    batched_prompt = [input_item['prompt'] for input_item in final_dataset]

    if '/' in config.get('model_name'):
        if MODEL is None:
            MODEL = LLM(model=config.get('model_name'), **config.get("model_args", {}))

        sampling_params = SamplingParams(**config.get('generation_args', {}))
        output_list = MODEL.generate(batched_prompt, sampling_params)
        
        results = []
        for input_id, output in zip(batched_ids, output_list):
            original_prompt = output.prompt
            generated_text = output.outputs[0].text
            if 'gpt-oss' in config.get('model_name'):
                parsed_response = parse_harmony_response(original_prompt, generated_text,
                                                        output.outputs[0].token_ids)
            else:
                parsed_response = parse_default_response(original_prompt, generated_text)

            results.append({
                "id": input_id,
                "reasoning_content": parsed_response.get("reasoning_content", None),
                "response": parsed_response.get("response", None)
            })
        
        return results
    else:
        results = []

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = []
            for input_id, msg in zip(batched_ids, batched_prompt):
                futures.append(
                    executor.submit(request_openai_completion, config.get('model_name'), input_id, msg)
                )

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating sessions"):
                results.append(fut.result())
        
        return results

def process_dataset_in_chunks(dataset_name, output_path, chunk_size, start_offset, end_offset, model_config,
                              safe_infer=False, debug=False):
    """Process the entire dataset in chunks"""
    total_size = 0
    if dataset_name in TRAIN_DATASETS_DICT_SIZE:
        total_size = TRAIN_DATASETS_DICT_SIZE[dataset_name]
    else:
        total_size = EVAL_DATASETS_DICT_SIZE[dataset_name]
        
    if end_offset > 0:
        total_size = min(total_size, end_offset)
    
    for offset in range(start_offset, total_size, chunk_size):
        logging.info(f"Processing offset: {offset}")
        
        chunk_dataset = create_prompt_dataset(
            dataset_name=dataset_name,
            output_path=output_path,
            model_config=model_config,
            chunk_size=chunk_size,
            offset=offset,
            safe_infer=safe_infer,
            debug=debug,
        )
        
        if chunk_dataset is not None and len(chunk_dataset) > 0:
            logging.info(f"Processing {len(chunk_dataset)} samples")
            
            # Generate responses for this chunk
            results = client_completion(model_config, chunk_dataset)
            write_results(results, save_path)
            
            # If debug mode, only process one chunk
            if debug:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on any evaluation datasets')
    parser.add_argument('--model_config_path', '-c', type=str, required=True,
                        help=f"Model's config for running evaluation. For example, see `data/eval_configs`.")
    parser.add_argument('--dataset_names', '-d', type=str, default="all",
                        help="List of dataset to be evaluated upon, separated by comma(s). `all` means infer on all.")
    parser.add_argument('--output_folder', '-o', type=str, default='output',
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

    # Create output folder
    output_folder = args.output_folder
    if not os.path.isabs(output_folder):
        output_folder = os.path.join(ROOT_DIR, args.output_folder)
        
    os.makedirs(output_folder, exist_ok=True)

    for seed in args.seeds_list:
        for dataset_name in eval_dataset_list: 
            if 'generation_args' not in model_config:
                model_config['generation_args'] = {}
            model_config['generation_args']['seed'] = seed
            save_name = f"{dataset_name}_{seed}.json"
            save_path = os.path.join(output_folder, save_name)
        
            if args.chunk_size > 0:
                process_dataset_in_chunks(
                    dataset_name=dataset_name,
                    output_path=save_path,
                    chunk_size=args.chunk_size,
                    start_offset=args.start_offset,
                    end_offset=args.end_offset,
                    model_config=model_config,
                    safe_infer=args.safe_infer,
                    debug=args.debug
                )
            else:
                process_dataset_in_chunks(
                    dataset_name=dataset_name,
                    output_path=save_path,
                    chunk_size=1000000,  # Large number to get all data at once
                    start_offset=args.start_offset,
                    end_offset=args.end_offset,
                    model_config=model_config,
                    safe_infer=args.safe_infer,
                    debug=args.debug
                )