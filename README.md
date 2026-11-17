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

`py -3.12 -m venv .venv`

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