# data_prep_simple.py

import os
import re
from collections import defaultdict

MIN_LEN = 2
MAX_LEN = 15
ONLY_LETTERS_RE = re.compile(r"^[a-z]+$")

def normalize_word(raw):
    w = raw.strip().lower()
    if not ONLY_LETTERS_RE.match(w):
        return None
    if not (MIN_LEN <= len(w) <= MAX_LEN):
        return None
    return w

def load_corpus_words(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            for tok in line.split():
                w = normalize_word(tok)
                if w is not None:
                    words.append(w)
    return words

def bucket_by_length(words):
    buckets = defaultdict(list)
    seen = defaultdict(set)
    for w in words:
        L = len(w)
        if w not in seen[L]:
            seen[L].add(w)
            buckets[L].append(w)
    # optional: sort deterministically
    for L in buckets:
        buckets[L].sort()
    return buckets


# Example usage:
if __name__ == "__main__":
    corpus_words = load_corpus_words("/Users/sudiksha/sudiksha/Acad/sem-5/machine-learning/hackathon/hangman-using-hmm-and-rl/datasets/corpus.txt")
    corpus_buckets = bucket_by_length(corpus_words)

    print("Corpus stats:")
    for L in range(MIN_LEN, MAX_LEN+1):
        print(f"Length {L}: {len(corpus_buckets.get(L, []))}")

    test_words = load_corpus_words("/Users/sudiksha/sudiksha/Acad/sem-5/machine-learning/hackathon/hangman-using-hmm-and-rl/datasets/test.txt")
    test_buckets = bucket_by_length(test_words)

    print("\nTest stats:")
    for L in range(MIN_LEN, MAX_LEN+1):
        print(f"Length {L}: {len(test_buckets.get(L, []))}")
