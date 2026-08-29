# ExTax: Explainable Disinformation Detection via Persuasion, Emotion, and Narrative Role Taxonomies

This repository contains an implementation of **ExTax**, a two-stage framework for explainable disinformation detection. ExTax first aligns text representations with interpretable taxonomic signals, then uses the aligned representation for binary fake/real classification.

The implementation is prepared for anonymous paper review. It intentionally does not include author names, affiliations, or non-anonymous project metadata.

## Overview

ExTax uses three complementary explanation taxonomies:

- **Persuasion taxonomy**: Attack on reputation, Justification, Simplification, Distraction, Call, and Manipulative wording.
- **Narrative role taxonomy**: Ethical Stabilizers, Altruistic Catalysts, Overt Aggressors, Deceptive Subversives, Institutional Toxins, and Marginalized Sufferers.
- **Emotion taxonomy**: Fear, Anger, Hope, Anxiety, and Sadness.

The model is trained in two stages:

1. **Stage 1: Feature alignment**
   The encoder output is trained to predict 17 interpretable features: 6 persuasion features, 6 narrative-role features, and 5 emotion features.

2. **Stage 2: Disinformation classification**
   The aligned feature representation is passed through a heterogeneous attention module and a classification head to predict `fake` or `real`.

## Repository Structure

```text
.
|-- config/
|   |-- prompts.py        # Prompt templates for feature annotation
|   |-- stage1.yaml       # Stage-1 feature alignment configuration
|   `-- stage2.yaml       # Stage-2 classification configuration
|-- data/
|   |-- raw_data/         # Original train/validation/test CSV files
|   |-- aug_data/         # LLM-generated taxonomy annotations
|   |-- dataset.py        # Dataset loading and feature parsing utilities
|   `-- preprocess.py     # LLM-based annotation script
|-- models/
|   `-- models.py         # Main ExTax model implementation
|-- train.py              # Training and evaluation entry point
|-- requirements.txt
`-- README.md
```

## Environment

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The project depends on PyTorch, Transformers, scikit-learn, pandas, NumPy, tqdm, and the OpenAI-compatible API client used for annotation.

The default configuration expects a local RoBERTa checkpoint at:

```text
../weights/roberta-large
```

If your checkpoint is stored elsewhere, update `model.encoder_name` in `config/stage1.yaml` and `config/stage2.yaml`. You can also use a Hugging Face model name such as `roberta-large` if network access and cache settings are available in your environment.

## Data Format

Raw data files are CSV files with at least the following columns:

```text
uuid,content,label
```

where `label` is one of:

```text
fake,real
```

Annotated files contain the original fields plus LLM-generated taxonomy outputs:

```text
uuid,content,label,article_type,system_prompt,user_prompt,generated_pred
```

The `generated_pred` column stores a JSON-like dictionary whose keys correspond to the taxonomy labels used by `data/dataset.py`.

The training code expects three annotation directories in this order:

```text
persuasion annotations
narrative-role annotations
emotion annotations
```

In this anonymized package, the included annotation files are under:

```text
data/aug_data/pcot_persu_data
data/aug_data/role_data
data/aug_data/emo_data
```

Before running training, make sure the `data.data_dirs` entries in the YAML config point to the actual annotation directories in your local copy.

## Running Experiments

### Stage 1: Feature Alignment

Run:

```bash
python train.py --config config/stage1.yaml
```

This stage trains the persuasion, role, and emotion feature heads. It reports macro F1 scores for each taxonomy group and saves the best checkpoint to the path configured by:

```yaml
save:
  save_path: "weights/extax_stage1_opt0_14.pth"
```

### Stage 2: Classification

After Stage 1 finishes, run:

```bash
python train.py --config config/stage2.yaml
```

This stage loads the Stage-1 checkpoint from `train.pretrained_path`, trains the classification components, and reports test macro F1.

If you use a different Stage-1 checkpoint path, update:

```yaml
train:
  pretrained_path: "weights/extax_stage1_opt0_14.pth"
```

## Annotation / Preprocessing

The repository also includes the script used to generate taxonomy annotations with OpenAI-compatible chat completion APIs:

```bash
python data/preprocess.py \
  --input data/raw_data/ECTF/train.csv \
  --output data/aug_data/pcot_persu_data/MODEL_ECTF_train.csv \
  --model MODEL_NAME \
  --mode persu \
  --max_workers 8
```

The `--mode` argument selects the annotation taxonomy:

```text
persu  # persuasion taxonomy
role   # narrative-role taxonomy
emo    # emotion taxonomy
```

API credentials and optional base URLs are loaded from environment variables, including:

```text
OPENAI_API_KEY
OPENAI_BASE_URL
DEEPSEEK_API_KEY
DEEPSEEK_BASE_URL
QWEN_API_KEY
QWEN_BASE_URL
CLAUDE_API_KEY_
CLAUDE_BASE_URL
GEMINI_API_KEY_
GEMINI_BASE_URL
```

## Configuration Notes

Important fields in the YAML files:

- `train.stage`: `align` for Stage 1 and `cls` for Stage 2.
- `train.device`: CUDA or CPU device string. Change this to match your machine, for example `cuda:0` or `cpu`.
- `data.batch_size`: Batch size for dataloaders.
- `data.max_length`: Maximum token length used by the tokenizer.
- `data.models`: Model names used for multi-annotator feature smoothing in Stage 1.
- `data.alpha`: Entropy-based smoothing strength for multi-model taxonomy labels.
- `model.num_hetero_attention`: Number of heterogeneous attention blocks used in classification.

## Outputs

Training prints validation and test metrics to standard output. If `save.save_path` is set, the trained checkpoint is saved as a PyTorch state dictionary.

Expected Stage-1 metrics:

```text
Test Persu F1
Test Role F1
Test Emo F1
```

Expected Stage-2 metric:

```text
Test Macro F1
```

## Reproducibility

The training script sets Python, NumPy, and PyTorch random seeds using the value in the YAML file:

```yaml
train:
  seed: 42
```

For exact reproducibility, use the same dependency versions, GPU type, CUDA/cuDNN settings, model checkpoint, data files, and YAML configuration.

## Citation

If you use this work, please cite:

```bibtex
@misc{luo2026extaxexplainabledisinformationdetection,
      title={ExTax: Explainable Disinformation Detection via Persuasion, Emotion, and Narrative Role Taxonomies}, 
      author={Shang Luo and Yingguang Yang and Zhenchen Sun and Yang Liu and Bin Chong and Jingru Chen and Yancheng Chen and Jiayu Liang and Kefu Xu and Hao Peng and Philip S. Yu},
      year={2026},
      eprint={2605.27045},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.27045}, 
}
```
