# Bilingual Customer Support Dataset

## Overview

This dataset was created specifically for the SWE015 Customer Support Ticket Router homework project.

It contains 310 English and Turkish customer messages related to home-appliance support and out-of-scope requests.

## Dataset Structure

File: `model_dataset_verified.csv`

Columns:
- `text`: Customer message
- `intent`: Ground-truth intent label
- `language`: Language (`en` or `tr`)
- `split`: Assigned dataset split

## Dataset Distribution

| Split | Messages |
|---|---:|
| Training | 206 |
| Validation | 46 |
| Test | 58 |
| **Total** | **310** |

Language distribution:
- English: 155
- Turkish: 155

Intent distribution:
- `technical_issue`: 20
- `repair_request`: 20
- `warranty_question`: 20
- `delivery_question`: 20
- `refund_request`: 20
- `out_of_scope`: 210

## Data Provenance

The messages were created specifically for this homework rather than copied from an external customer-support dataset.

## Evaluation Notes

The dataset contains a relatively high proportion of out-of-scope messages (210/310). Consequently, overall accuracy should be interpreted alongside macro-F1, in-scope accuracy, and out-of-scope recall.

The predefined `split` column should be used to preserve the original train, validation, and test partitions.

## Reproducibility

The CSV is provided to support inspection and reproduction of the project's training and evaluation experiments.

The original training scripts reference `data/model_dataset_verified.csv`. To use the published dataset with those scripts, copy the CSV from `dataset/` into `data/` without modifying its contents.

Model artifacts for running the application are hosted separately on Hugging Face.