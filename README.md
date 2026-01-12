# CodeMixQA
![Pull Requests Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<a href="https://huggingface.co/datasets/gentaiscool/codemixqa">
  <img src="https://img.shields.io/badge/CodeMixQA Dataset-orange.svg?logo=huggingface&logoColor=white" alt="CodeMixQA Dataset"/>
</a>
  
A benchmark with high-quality human annotations, comprising 16 diverse parallel code-switched language-pair variants that span multiple geographic regions and code-switching patterns, and include both original scripts and their transliterated forms.

We use SimpleQA Verified as our source dataset. We select the SimpleQA Verified, as it is a challenging evaluation set that has not been saturated yet by current models and has desirable properties such as verifiable answers (through source reconciliation), de-duplicated data points, topic balancing, and that it is markedly different from most standard tasks that are prevalent in code switching studies such as language identification, NER, and machine translation.

In this dataset, we employ multiple data generation strategies, including random switching, selective switching, and grammar-constrained approaches. This dataset enables systematic evaluation of LLM performance across different code-switching patterns and text generation strategies.

## 🧪 Generate Dataset

```
python generate_codemix_data.py
```

### Main Arguments
| Argument         | Description                                       | Example / Default                     |
|------------------|---------------------------------------------------|---------------------------------------|
