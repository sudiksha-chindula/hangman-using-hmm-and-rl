# inference.py
import pickle
import numpy as np
from utils_hmm import letter_to_idx

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def load_hmm_for_length(L):
    """
    Load the saved HMM for a given word length L.
    """
    path = f"hmms/hmm_len{L}.pkl"
    with open(path, "rb") as f:
        hmm = pickle.load(f)
    return hmm


def pattern_to_index_seq(pattern):
    """
    Convert "_a__e" -> [None, 0, None, None, 4]
    """
    seq = []
    for c in pattern.lower():
        if c == "_":
            seq.append(None)
        else:
            seq.append(letter_to_idx(c))
    return seq


def predict_letter_probs(pattern, guessed_wrong, guessed_right=None):
    """
    Input:
        pattern: string like "_a__e"
        guessed_wrong: set of letters known to be incorrect
        guessed_right: optional set of letters correctly guessed so far

    Output:
        A probability distribution (numpy array of size 26)
        over letters a-z for all unknown positions in pattern.
    """
    if guessed_right is None:
        guessed_right = set()

    seq = pattern_to_index_seq(pattern)
    L = len(seq)

    hmm = load_hmm_for_length(L)
    K, V, T = hmm.K, hmm.V, L

    wrong_idxs = set(letter_to_idx(c) for c in guessed_wrong)
    right_idxs = set(letter_to_idx(c) for c in guessed_right)

    # ---------------------------------------------------------
    # Build masked emission table Bmask for each state
    # ---------------------------------------------------------
    Bmask = np.copy(hmm.B)

    for t in range(T):
        if seq[t] is not None:
            # Known letter → only that letter allowed
            correct_letter = seq[t]
            for s in range(K):
                for v in range(V):
                    if v != correct_letter:
                        Bmask[s, v] = 0.0
                if Bmask[s].sum() == 0:
                    Bmask[s] = hmm.B[s]
                Bmask[s] /= Bmask[s].sum()
        else:
            # Unknown position → drop wrong letters
            for s in range(K):
                for v in wrong_idxs:
                    Bmask[s, v] = 0.0
                if Bmask[s].sum() == 0:
                    Bmask[s] = hmm.B[s]
                Bmask[s] /= Bmask[s].sum()

    # ---------------------------------------------------------
    # Forward (scaled)
    # ---------------------------------------------------------
    alpha = np.zeros((T, K))
    scale = np.zeros(T)

    for s in range(K):
        if seq[0] is not None:
            emit = Bmask[s, seq[0]]
        else:
            emit = Bmask[s].sum()
        alpha[0, s] = hmm.pi[s] * emit

    scale[0] = alpha[0].sum()
    alpha[0] /= scale[0]

    for t in range(1, T):
        for j in range(K):
            if seq[t] is not None:
                emit = Bmask[j, seq[t]]
            else:
                emit = Bmask[j].sum()
            alpha[t, j] = np.dot(alpha[t-1], hmm.A[:, j]) * emit

        scale[t] = alpha[t].sum()
        alpha[t] /= scale[t]

    # ---------------------------------------------------------
    # Backward (scaled)
    # ---------------------------------------------------------
    beta = np.zeros((T, K))
    beta[-1] = 1.0 / scale[-1]

    for t in range(T-2, -1, -1):
        for i in range(K):
            total = 0.0
            for j in range(K):
                if seq[t+1] is not None:
                    emit = Bmask[j, seq[t+1]]
                else:
                    emit = Bmask[j].sum()
                total += hmm.A[i, j] * emit * beta[t+1, j]
            beta[t, i] = total / scale[t]

    # ---------------------------------------------------------
    # Posterior letter probabilities over all blank positions
    # ---------------------------------------------------------
    letter_probs = np.zeros(V)

    for t in range(T):
        if seq[t] is None:  # blank position
            for v in range(V):
                if v in wrong_idxs or v in right_idxs:
                    continue
                p = 0.0
                for s in range(K):
                    p += alpha[t, s] * hmm.B[s, v] * beta[t, s]
                letter_probs[v] += p

    if letter_probs.sum() > 0:
        letter_probs /= letter_probs.sum()

    return letter_probs


def recommend_next_letter(pattern, guessed_wrong, guessed_right=None):
    """
    Returns the single best next guess (highest probability letter).
    """
    probs = predict_letter_probs(pattern, guessed_wrong, guessed_right)
    best_idx = np.argmax(probs)
    return LETTERS[best_idx]


def top_k_letters(pattern, guessed_wrong, guessed_right=None, k=5):
    """
    Returns a list of top-k (letter, probability) pairs.
    """
    probs = predict_letter_probs(pattern, guessed_wrong, guessed_right)
    idxs = np.argsort(probs)[::-1]
    return [(LETTERS[i], probs[i]) for i in idxs[:k]]


# ---------------------------------------------------------
# TEST BLOCK (RUN DIRECTLY)
# ---------------------------------------------------------
if __name__ == "__main__":
    # Example test:
    pattern = "_a__e"
    guessed_wrong = {'r', 't', 'o'}
    guessed_right = {'a', 'e'}

    print(f"Pattern: {pattern}")
    print(f"Wrong guesses: {guessed_wrong}")
    print(f"Right guesses: {guessed_right}")

    next_letter = recommend_next_letter(pattern, guessed_wrong, guessed_right)
    print(f"\nRecommended next guess: {next_letter}")

    print("\nTop-5 suggested letters:")
    for letter, p in top_k_letters(pattern, guessed_wrong, guessed_right, k=5):
        print(f"{letter}: {p:.4f}")
