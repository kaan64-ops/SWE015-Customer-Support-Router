# Third-Party Model and Embedding Notices

This project uses pretrained resources developed by third parties.

## Multilingual BERT

- Base model: `google-bert/bert-base-multilingual-cased`
- Source: https://huggingface.co/google-bert/bert-base-multilingual-cased
- License: Apache License 2.0
- Usage: Fine-tuned for bilingual customer support intent classification.

## Multilingual DistilBERT

- Base model: `distilbert/distilbert-base-multilingual-cased`
- Source: https://huggingface.co/distilbert/distilbert-base-multilingual-cased
- License: Apache License 2.0
- Usage: Student Transformer for knowledge distillation and INT8 quantization.

## fastText Aligned Word Vectors

- Source: https://fasttext.cc/docs/en/aligned-vectors
- License: Creative Commons Attribution-ShareAlike 3.0
- Usage: English and Turkish aligned word embeddings, with a compact subset used for feedforward neural network inference.
- Attribution: Joulin et al. (2018), *Loss in Translation: Learning Bilingual Word Mapping with a Retrieval Criterion*; Bojanowski et al. (2017), *Enriching Word Vectors with Subword Information*.

The pretrained resources retain their respective third-party licensing conditions. The project's own source-code licensing is a separate matter.