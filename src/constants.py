import os
from datetime import date

try:
    from openai_harmony import (
        SystemContent,
        ReasoningEffort,
        HarmonyEncodingName,
        load_harmony_encoding,
    )
    # You can set a flag to indicate that the library is enabled
    OPENAI_ENABLED = True
    HARMONY_ENCODING = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    SYSTEM_MESSAGE_OSS = (
        SystemContent.new()
        .with_model_identity("You are ChatGPT, a large language model trained by OpenAI.")
        .with_reasoning_effort(ReasoningEffort.LOW)
        .with_conversation_start_date(date.today().isoformat())
        .with_knowledge_cutoff("2024-06")
        .with_required_channels(["analysis", "commentary", "final"])
    )
except ImportError:
    # If the import fails, set the flag to false
    OPENAI_ENABLED = False

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.dirname(CUR_DIR))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

CLIENT_RETRIES = 3
RANDOM_SEED = 42
DEBUG_COUNT = 10

TRAIN_DATASETS_DICT = {
}

TRAIN_DATASETS_DICT_SIZE = {
}

EVAL_DATASETS_DICT = {
    "simpleqa-xl": "davidanugraha/simpleqa-verified-XL",
    "simpleqa": "davidanugraha/simpleqa-verified"
}

EVAL_DATASETS_DICT_SIZE = {
    "simpleqa-xl": 72000,
    "simpleqa": 1000
}