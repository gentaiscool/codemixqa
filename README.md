# CodeMixQA
![Pull Requests Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<a href="https://huggingface.co/datasets/gentaiscool/codemixqa">
  <img src="https://img.shields.io/badge/CodeMixQA Dataset-orange.svg?logo=huggingface&logoColor=white" alt="CodeMixQA Dataset"/>
</a>
  
A benchmark with high-quality human annotations, comprising 16 diverse parallel code-switched language-pair variants that span multiple geographic regions and code-switching patterns, and include both original scripts and their transliterated forms.

We use SimpleQA Verified as our source dataset. We select the SimpleQA Verified, as it is a challenging evaluation set that has not been saturated yet by current models and has desirable properties such as verifiable answers (through source reconciliation), de-duplicated data points, topic balancing, and that it is markedly different from most standard tasks that are prevalent in code switching studies such as language identification, NER, and machine translation.

In this dataset, we employ multiple data generation strategies, including random switching, selective switching, and grammar-constrained approaches. This dataset enables systematic evaluation of LLM performance across different code-switching patterns and text generation strategies.

## 📜 Paper 
This is the source code of the paper [[Arxiv (will be updated)]](link). This code has been written using Python. If you use any code or datasets from this toolkit in your research, please cite the associated paper.
```bibtex
Will be updated
```

### ⚡ Environment Setup
Please run the following command to install the required libraries to reproduce the benchmark results.
#### Via `pip`
```
pip install -r requirements.txt
```

## 🧪 Generate Dataset
This is the command to generate code-switched dataset.
```
python generate_codemix_data.py --openai_key <OPENAI_KEY>
```

### Arguments
| Argument         | Description                                       | Example / Default                     |
|------------------|---------------------------------------------------|---------------------------------------|
| `--openai_key`   | OPENAI_KEY                                        | sk-....                               |

## 🧪 Generate Transliteration
The transliteration is only done for Indic languages.
```
python generate_transliterated_data.py --openai_key <OPENAI_KEY> --language <LANGUAGE>
```

### Arguments
| Argument         | Description                                       | Example / Default                     |
|------------------|---------------------------------------------------|---------------------------------------|
| `--openai_key`   | OPENAI_KEY                                        | sk-....                               |
| `--language`     | Language                                          | Hindi                                 |

## 🧪 Run Evaluation

```
python src/inference.py -d <DATASET_NAMES> -o <OUTPUT>
```

### Arguments
| Argument         | Description                                       | Example / Default                     |
|------------------|---------------------------------------------------|---------------------------------------|
| `--dataset_names` or `-d` | Dataset names                                     | all                                   |
| `--output_folder` or `o` | Output folder                                     | output                                |
| `--chunk_size`| Output folder                                     | 1                                |
| `--start_offset`| Start offset                                     | 0                                |
| `--end_offset`| End offset                                     | -1                                |
| `--seeds_list`| List of seeds to use. Provide one or more integers separated by spaces (e.g., --seeds_list 0 1 2). Defaults to [0, 1, 2].                                     | 0 1 2                                |
| `--safe-infer`| Filter out input that is longer than max-model-len minus output length                                     | (store_true)                                 |
| `--debug`| Debug with {DEBUG_COUNT} samples.                 | (store_true)                                |
