import json
import os
import logging
from datasets import Dataset, load_dataset

from .constants import *
from .utils import *

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

# ---------------------------------------------------------------------
# Base Class
# ---------------------------------------------------------------------
class GenericDataset:
    def __init__(self, model_config, output_path, debug=False, surgery=False):
        self.output_path = output_path
        self.debug = debug 
        self.surgery = surgery
        self.tokenizer = None
        self.formatting = "default"
        
        if "gpt-oss" in model_config.get("model_name", ""):
            self.formatting = "gpt-oss"
        elif "gpt-5" in model_config.get("model_name", "") or "gpt-4" in model_config.get("model_name", ""):
            self.formatting = "openai"
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_config.get("model_name"))
            if "gemma" in model_config.get("model_name", "").lower() or "llama" in model_config.get("model_name", "").lower():
                self.formatting = "non-thinking"
            if "olmo" in model_config.get("model_name", "").lower():
                self.formatting = "olmo-thinking"

    def _get_existing_ids(self):
        """Read the output file and return a set of existing IDs."""
        existing_ids = set()
        if os.path.exists(self.output_path):
            with open(self.output_path, 'r') as f:
                for obj in json.load(f):
                    if not self.surgery:
                        # Don't care, still add regardless
                        existing_ids.add(obj['id'])
                    elif obj['response'] is not None:
                        try:
                            if not (obj['response'] is None or obj['reasoning_content'] is None or obj['response'] == "" or obj['reasoning_content'] == ""):
                                existing_ids.add(obj['id'])
                            if obj["response"].strip().endswith("</answer>"):
                                existing_ids.add(obj['id'])
                        except Exception as e:
                            # Means need to be fixed, so no need to add to existing
                            pass
                        
        return existing_ids

    def _get_initial_dataset(self, dataset_id, offset, chunk_size, subset='default', split='train'):
        """Loads dataset chunk for processing."""
        existing_ids = self._get_existing_ids()   
        
        if dataset_id.endswith(".json") or dataset_id.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=dataset_id, streaming=True, split="train")
        elif subset == "default":
            dataset = load_dataset(dataset_id, streaming=True, split=split)
        else:
            dataset = load_dataset(dataset_id, subset, streaming=True, split=split)

        # Skip to the offset position
        if offset > 0:
            dataset = dataset.skip(offset)
        
        # Take only chunk_size items
        dataset = dataset.take(chunk_size)
        
        # Filter out existing IDs
        if len(existing_ids) > 0:
            dataset = dataset.filter(lambda example: example["id"] not in existing_ids)
        
        if self.debug:
            dataset = dataset.take(min(DEBUG_COUNT, chunk_size))
        
        # Convert the chunk to a regular dataset
        chunk_data = list(dataset)
        if len(chunk_data) == 0:
            return None
        
        chunk_dataset = Dataset.from_list(chunk_data)
        
        return chunk_dataset

    def get_prompt_dataset(self, dataset_id, split, chunk_size, offset):
        dataset = self._get_initial_dataset(dataset_id=dataset_id,
                                            offset=offset,
                                            chunk_size=chunk_size,
                                            split=split)
        if dataset is None or len(dataset) == 0:
            return dataset

        dataset = dataset.map(lambda row: {
            "id": row.get("id"),
            "prompt": self.build_conversation(row)
        }, num_proc=8)

        return dataset

    def get_final_prompt(self, developer_text, user_text):
        if self.formatting == 'gpt-oss':
            from openai_harmony import Conversation, Message, Role
            system_msg = SYSTEM_MESSAGE_OSS
            convo = Conversation.from_messages(
                [
                    Message.from_role_and_content(Role.SYSTEM, system_msg),
                    Message.from_role_and_content(Role.DEVELOPER, developer_text),
                    Message.from_role_and_content(Role.USER, user_text),
                ]
            )
            input_tokens = HARMONY_ENCODING.render_conversation_for_completion(convo, Role.ASSISTANT)
            final_prompt =  HARMONY_ENCODING.decode(input_tokens)

            return final_prompt
        elif self.formatting == 'openai':
            return [{'role': 'user', 'content': user_text}]
        else:
            convo = [{'role': 'system', 'content': f"{developer_text}"},
                    {'role': 'user', 'content': user_text}]

            if self.formatting == "non-thinking":
                final_prompt = self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            elif self.formatting == "olmo-thinking":
                final_prompt = self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            else:
                final_prompt = self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True, enable_thinking=True)
                if "<think>" not in final_prompt:
                    final_prompt += "<think>"
                    
            return final_prompt

# ---------------------------------------------------------------------
# General Open Ended Dataset
# ---------------------------------------------------------------------
class GeneralOpenEndedDataset(GenericDataset):
    def __init__(self, model_config, output_path, debug=False, surgery=False):
        super().__init__(model_config, output_path, debug, surgery)

    def build_conversation(self, row):
        # Developer message
        developer_text = f"""You are an expert reasoning model. Your task is to carefully read and answer the given question.
Explain your reasoning clearly and logically before presenting your final answer.
"""
        # User message
        user_text = row['problem']

        return self.get_final_prompt(developer_text, user_text)

def create_prompt_dataset(dataset_name, output_path, model_config,
                          chunk_size, offset, safe_infer=False,
                          surgery=False, debug=False):

    # Determine split and dataset ID
    if dataset_name in TRAIN_DATASETS_DICT:
        use_split = "train"
        dataset_id = TRAIN_DATASETS_DICT[dataset_name]
    elif dataset_name in EVAL_DATASETS_DICT:
        use_split = "test"
        dataset_id = EVAL_DATASETS_DICT[dataset_name]
    else:
        raise NotImplementedError(f"Dataset `{dataset_name}` not implemented!")

    # Map dataset type
    if dataset_name.lower() in ["simpleqa-xl", "simpleqa"]:
        dataset_cls = GeneralOpenEndedDataset(model_config, output_path, debug, surgery)
    elif dataset_name.startswith("judge"):
        dataset_cls = JudgeDataset(model_config, output_path, debug, surgery)
    else:
        raise NotImplementedError(f"Other dataset `{dataset_name}` not yet supported!")

    # Build dataset
    dataset_chunk = dataset_cls.get_prompt_dataset(dataset_id=dataset_id,
                                                   split=use_split,
                                                   chunk_size=chunk_size,
                                                   offset=offset)

    # Safe inference filtering
    if safe_infer and dataset_chunk is not None:
        original_len = len(dataset_chunk)
        safe_input_len = model_config.get("model_args", {}).get("max_model_len", 32768) - \
                         model_config.get("generation_args", {}).get("max_tokens", 8192)
        dataset_chunk = dataset_chunk.filter(lambda row: len(row["prompt"]) < safe_input_len, num_proc=8)
        logging.info(f"Safe infer enabled! {original_len} → {len(dataset_chunk)} samples kept.")

    return dataset_chunk
