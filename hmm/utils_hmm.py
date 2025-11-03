# utils_hmm.py

def letter_to_idx(c):
    """Convert a lowercase letter 'a'-'z' to an integer 0-25."""
    return ord(c) - ord('a')


def word_to_idx_seq(word):
    """
    Convert a word into a list of integers.
    Example:
      'cat' -> [2, 0, 19]
    """
    return [letter_to_idx(c) for c in word]
