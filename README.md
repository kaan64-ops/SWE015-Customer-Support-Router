# Bilingual Customer Support Ticket Router

## Project Overview

This project is an intelligent customer support ticket routing application for home appliances. It supports customer messages in **English and Turkish**.

The application:
- Normalizes incoming messages.
- Extracts product model codes using regular expressions.
- Corrects misspelled appliance names using Levenshtein edit distance.
- Responds to common greetings and small talk using ELIZA-style rules.
- Predicts customer support intents.
- Routes low-confidence and out-of-scope messages to a human support queue.
- Compares six model predictions side by side.

## Intent Categories

1. Delivery Question
2. Refund Request
3. Repair Request
4. Technical Issue
5. Warranty Question
6. Out of Scope

## Models

The application includes six inference options:

1. Rule-Based Baseline
2. TF-IDF + Logistic Regression
3. Feedforward Neural Network with fastText Embeddings
4. Fine-Tuned Multilingual BERT
5. Distilled Student Transformer
6. INT8 Quantized Student Transformer

The four-model experimental ladder consists of models 2–5. The rule-based baseline and quantized student provide additional comparisons.

## Environment

Developed and tested on Windows 11 with Python 3.12.14.

## Installation

Clone the repository and open its root folder.

Create a virtual environment:

Install Python 3.12 first and ensure it is available from your terminal. On Windows, py -3.12 -m venv .venv is an alternative when the Python launcher is installed.

`python -m venv .venv`

Activate it in PowerShell:

`.\.venv\Scripts\Activate.ps1`

Install the required libraries:

`python -m pip install -r requirements.txt`

## Model Artifacts

Trained models and compact embeddings are distributed separately because of their size.

**Model download:** [Download inference_artifacts.zip from Hugging Face](https://huggingface.co/KaanCN64/customer-support-router/resolve/main/inference_artifacts.zip)

**Hugging Face model repository:** https://huggingface.co/KaanCN64/customer-support-router

Download `inference_artifacts.zip` and extract it into the project root.

The archive contains the required `models/` and `embeddings/` folders.

Only load model artifacts obtained from the project author or another trusted source.

## Run the Application

With the virtual environment activated, run this command from the project root:

`python -m streamlit run app.py`

Enter an English or Turkish customer message, select a model, and inspect its predicted intent, confidence score, routing decision, and side-by-side model comparison.

## Evaluation

Evaluation results are available in the `results/` folder.

Reported metrics include:
- Accuracy
- Macro-F1
- In-scope accuracy
- Out-of-scope recall
- Model size
- CPU inference latency

The latency benchmark used 120 timed inference runs per model after 10 warm-up runs.

## Limitations

The held-out dataset is small and contains more out-of-scope than in-scope examples. Overall accuracy therefore does not establish production-level reliability.

Model confidence scores are not calibrated probabilities.

The INT8 quantized student is smaller than the

## AI Usage

AI tools assisted with implementation, debugging, and documentation. The final presentation will disclose the tools used and their roles.

## Training and Reproducibility

The original training and evaluation scripts are provided in `src/`.

The verified bilingual dataset is available at `dataset/model_dataset_verified.csv`. Training scripts expect it at `data/model_dataset_verified.csv`.

### Preparing the Dataset

From the project root, run:

```powershell
New-Item -ItemType Directory -Path .\data -Force
Copy-Item .\dataset\model_dataset_verified.csv .\data\model_dataset_verified.csv
```

### Training Sequence

The following commands describe the model-training workflow:

```powershell
python src/train_rule_baseline.py
python src/train_classical_baseline.py
python src/train_feedforward.py
python src/train_multilingual_bert.py
python src/train_distilled_student.py
python src/quantize_distilled_student.py
python src/validate_quantized_student.py
```

The feedforward model uses `embeddings/bilingual_compact_vectors.npz`, which is included in the downloadable inference artifacts on Hugging Face. The original extraction script, `src/extract_fasttext_vectors.py`, uses pretrained English and Turkish aligned fastText `.vec` files. These large source embedding files are not included in the GitHub repository.

The distilled student requires the fine-tuned multilingual BERT teacher checkpoint. The INT8 quantization step requires the trained distilled student checkpoint.

### Reproducing the Evaluation

The original evaluation and benchmark scripts are:

```powershell
python src/evaluate_final_test.py
python src/benchmark_cpu_latency.py
```

Published evaluation results are available in `results/`.

**Reproducibility note:** The application and all six inference models were successfully tested in a fresh GitHub clone. The full training sequence has not been independently rerun from scratch. Training may require downloading pretrained models, obtaining the compact embeddings, and additional computation. Reported results are from the original completed experiments.