# BitNet Summarization Experiment (XSUM + XL-Sum Subset)

## Overview

This repository implements **Experiment 2** from the BitNet evaluation thesis and provides a framework to fine-tune and evaluate the following models on summarization tasks:

- **BitNet b1.58** – a 2B-parameter model with native 1-bit quantization  
- Baselines: **BART-large**, **mBART-50**, **Gemma-2B**, **GPT‑NeoX-2.7B**  
- Optional extractive reference: **DistilBERT**

⚠️ **Important Note on BitNet Efficiency**: While this experiment uses the transformers library for research purposes, please note that the full computational efficiency benefits of BitNet (speed, latency, energy consumption) are only available through the dedicated C++ implementation [bitnet.cpp](https://github.com/microsoft/bitnet.cpp). The transformers implementation will primarily demonstrate memory savings but may not show the performance improvements described in the technical paper.

Two summarization tasks are supported:

1. **XSUM** – English abstractive summarization from BBC news (one-sentence target)  
2. **XL-Sum** – multilingual summarization on six languages: English, Spanish, Hindi, Amharic, Sinhala, and Hausa

The goal is a systematic comparison of generative quality across models while enabling fast iteration via subsampling.

---

## Motivation and Research Objectives

- Evaluate BitNet’s summarization quality relative to strong full-precision baselines in English (RQ1).  
- Investigate BitNet’s performance on multilingual summarization with a balanced subset of high- and low-resource languages (RQ2, RQ3).  
- Assess the impact of extreme quantization on model performance, particularly for low-resource languages.

---

## Key Features

- Clear separation of training (`train.py`) and evaluation (`evaluate.py`) workflows  
- Hyperparameters managed through `config.yaml` with CLI overrides  
- Supports mixed precision, gradient checkpointing, and controlled data subsampling  
- Outputs standardized ROUGE-1, ROUGE-2, ROUGE-L scores per dataset and language

---

## Setup

1. Create and activate a Python 3.9+ virtual environment:
```bash
    python3 -m venv venv
    source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install BitNet-compatible Transformers fork:
```bash
pip install git+https://github.com/huggingface/transformers.git@096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4
```

## Configuration
All key parameters are stored in `config.yaml`, which includes model definitions, dataset languages, training hyperparameters, and generation settings. These can be overridden via command-line flags to enable rapid experimentation.

## Usage Examples

### Training (XSUM)
```bash
python src/train.py \
  --model bitnet \
  --model_name microsoft/bitnet-b1.58-2B-4T-bf16 \
  --dataset xsum \
  --batch_size 2 \
  --lr 1e-4 \
  --epochs 2 \
  --max_train 1000 \
  --max_val 100 \
  --output outputs/bitnet_xsum
```

### Evaluation (XSUM)
```bash
python src/evaluate.py \
  --model outputs/bitnet_xsum \
  --dataset xsum \
  --batch_size 4 \
  --beam 4 \
  --max_gen_len 60
```

### Multilingual Mode (XL-Sum)
```bash
python src/train.py \
  --model mbart \
  --model_name facebook/mbart-large-50 \
  --dataset xlsum \
  --languages en,es,hi,am,si,ha \
  --batch_size 1 \
  --lr 2e-4 \
  --epochs 1 \
  --max_train 1200 \
  --output outputs/mbart_xlsum6

python src/evaluate.py \
  --model outputs/mbart_xlsum6 \
  --dataset xlsum \
  --languages en,es,hi,am,si,ha \
  --batch_size 2 \
  --beam 4 \
  --max_gen_len 80
```

## Expected Outcomes
- BitNet should yield ~80–90% of baseline ROUGE scores on XSUM, with competitive performance after full training
- In multilingual setting, mBART is expected to lead overall; BitNet and GPT‑NeoX are competitive on high-resource languages but decline in low-resource
- Quantization effects are most pronounced in low-resource languages, confirming hypotheses regarding RQ3

## Citations and Licensing
- BitNet b1.58 (MSR, MIT License)
- Datasets: XSUM, XL-Sum (CC-BY-NC-SA 4.0)
- Baselines: BART, mBART, Gemma, GPT-NeoX
- Experiment code: Apache-2.0 license
