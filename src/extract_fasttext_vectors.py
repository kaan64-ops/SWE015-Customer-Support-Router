
from pathlib import Path
import re
import numpy as np
import pandas as pd

DATA_PATH = Path("data/model_dataset_verified.csv")
EMBEDDING_DIR = Path("embeddings")
OUTPUT_PATH = EMBEDDING_DIR / "bilingual_compact_vectors.npz"

DIM = 300

TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)

def tokenize(text):
    return TOKEN_PATTERN.findall(str(text).casefold())

data = pd.read_csv(
    DATA_PATH,
    dtype=str,
    encoding="utf-8-sig"
).fillna("")

# Vocabulary comes from train + validation only.
development = data[
    data["split"].isin(["train", "validation"])
].copy()

vocabularies = {}

for language in ["en", "tr"]:
    texts = development.loc[
        development["language"] == language, "text"
    ]

    vocabulary = set()

    for text in texts:
        vocabulary.update(tokenize(text))

    vocabularies[language] = vocabulary

    print(
        f"{language.upper()} vocabulary: "
        f"{len(vocabulary)} unique tokens"
    )

def extract_vectors(path, vocabulary, language):
    found = {}

    print(f"\nScanning {path.name} ...", flush=True)

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace"
    ) as file:

        first_line = file.readline().strip().split()

        # Some .vec files begin with a header:
        # vocabulary_size embedding_dimension
        has_header = (
            len(first_line) == 2
            and all(part.isdigit() for part in first_line)
        )

        if not has_header:
            file.seek(0)

        for line_number, line in enumerate(file, start=1):
            parts = line.rstrip().split(" ", 1)

            if len(parts) != 2:
                continue

            word, vector_text = parts
            word = word.casefold()

            if word not in vocabulary or word in found:
                continue

            vector = np.fromstring(
                vector_text,
                sep=" ",
                dtype=np.float32
            )

            if vector.size != DIM:
                continue

            found[word] = vector

            if len(found) == len(vocabulary):
                break

            if line_number % 500000 == 0:
                print(
                    f"Scanned {line_number:,} lines; "
                    f"found {len(found)} vectors",
                    flush=True
                )

    coverage = (
        len(found) / len(vocabulary)
        if vocabulary else 0
    )

    print(
        f"{language.upper()} coverage: "
        f"{len(found)}/{len(vocabulary)} "
        f"({coverage:.1%})"
    )

    return found

english = extract_vectors(
    EMBEDDING_DIR / "wiki.en.align.vec",
    vocabularies["en"],
    "en"
)

turkish = extract_vectors(
    EMBEDDING_DIR / "wiki.tr.align.vec",
    vocabularies["tr"],
    "tr"
)

# Prefix tokens with language to prevent collisions
# between identically spelled English/Turkish words.
tokens = []
vectors = []

for language, found in [("en", english), ("tr", turkish)]:
    for word in sorted(found):
        tokens.append(f"{language}:{word}")
        vectors.append(found[word])

if not vectors:
    raise ValueError("No matching embeddings were found.")

matrix = np.stack(vectors).astype(np.float32)

np.savez_compressed(
    OUTPUT_PATH,
    tokens=np.array(tokens),
    vectors=matrix
)

print("\nCOMPACT EMBEDDING EXTRACTION COMPLETE")
print("=" * 55)
print("Extracted tokens:", len(tokens))
print("Embedding matrix shape:", matrix.shape)
print("Saved:", OUTPUT_PATH)
print(
    "File size (MB):",
    round(OUTPUT_PATH.stat().st_size / (1024 ** 2), 2)
)
