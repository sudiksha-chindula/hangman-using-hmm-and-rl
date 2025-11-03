
# train_hmms.py

import os
import pickle
import numpy as np
from hmm_model import DiscreteHMM
from data_prep import load_corpus_words, bucket_by_length
from utils_hmm import word_to_idx_seq

# Hyperparameters
K_STATES = 10        # hidden states
MAX_ITERS = 20       # EM iterations
SAVE_DIR = "hmms"    # folder to store hmm_lenL.pkl

def train_hmm_for_length(words, L):
    """
    Train a DiscreteHMM for words of length L.
    words = list of strings (e.g. ["apple", "apply", ...])
    """
    if len(words) == 0:
        return None

    # Convert each word to a sequence of integer observations
    sequences = [word_to_idx_seq(w) for w in words]

    # Create HMM
    hmm = DiscreteHMM(K=K_STATES, V=26)

    # EM (Baum–Welch)
    hmm.baum_welch(sequences, max_iters=MAX_ITERS)

    return hmm


def main():
    # 1) Load training corpus
    corpus_words = load_corpus_words("/Users/sudiksha/sudiksha/Acad/sem-5/machine-learning/hackathon/hangman-using-hmm-and-rl/datasets/corpus.txt")
    buckets = bucket_by_length(corpus_words)

    # 2) Make sure output dir exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 3) Train HMM for each length
    for L, words in buckets.items():
        print(f"\n=== Training HMM for length {L} (n={len(words)}) ===")

        hmm = train_hmm_for_length(words, L)
        if hmm is None:
            print(f"Skipping length {L} (no data).")
            continue

        # Save model
        out_path = os.path.join(SAVE_DIR, f"hmm_len{L}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(hmm, f)
        print(f"Saved HMM for length {L} to {out_path}")


if __name__ == "__main__":
    main()
