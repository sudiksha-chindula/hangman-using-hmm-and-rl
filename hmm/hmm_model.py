# hmm_model.py
import numpy as np
import pickle

class DiscreteHMM:
    """
    A simple discrete-emission HMM:
        - K hidden states
        - 26 discrete observations (letters a-z)
    """

    def __init__(self, K=10, V=26, seed=42):
        """
        K = number of hidden states
        V = vocabulary size (26 letters)
        """

        self.K = K      # hidden states
        self.V = V      # observation symbols

        rng = np.random.default_rng(seed)

        # Initial state probabilities π (size K)
        # Random init but normalized
        self.pi = rng.random(K)
        self.pi /= self.pi.sum()

        # Transition matrix A (K x K)
        # A[i,j] = P(next hidden state = j | current state = i)
        self.A = rng.random((K, K))
        self.A /= self.A.sum(axis=1, keepdims=True)

        # Emission matrix B (K x V)
        # B[i, v] = P(observation=v | hidden state=i)
        self.B = rng.random((K, V))
        self.B /= self.B.sum(axis=1, keepdims=True)


    # ---------------------------------------------------------
    # Forward pass (scaled) → returns alpha and scaling factors
    # ---------------------------------------------------------
    def _forward(self, seq):
        T = len(seq)
        alpha = np.zeros((T, self.K))
        scale = np.zeros(T)

        # init
        alpha[0] = self.pi * self.B[:, seq[0]]
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]

        # recursive
        for t in range(1, T):
            for j in range(self.K):
                alpha[t, j] = np.dot(alpha[t-1], self.A[:, j]) * self.B[j, seq[t]]
            scale[t] = alpha[t].sum()
            alpha[t] /= scale[t]

        return alpha, scale


    # ---------------------------------------------------------
    # Backward pass (scaled)
    # ---------------------------------------------------------
    def _backward(self, seq, scale):
        T = len(seq)
        beta = np.zeros((T, self.K))

        # init
        beta[-1] = 1.0 / scale[-1]

        # recursive
        for t in range(T-2, -1, -1):
            for i in range(self.K):
                beta[t, i] = np.sum(self.A[i] * self.B[:, seq[t+1]] * beta[t+1])
            beta[t] /= scale[t]

        return beta


    # ---------------------------------------------------------
    # Baum–Welch training (EM)
    # ---------------------------------------------------------
    def baum_welch(self, sequences, max_iters=10):
        """
        sequences: list of lists, each seq is list of ints [0..25]
        """

        N = len(sequences)
        if N == 0:
            return

        for it in range(max_iters):
            # Expected counts
            pi_acc = np.zeros(self.K)
            A_acc = np.zeros((self.K, self.K))
            B_acc = np.zeros((self.K, self.V))
            log_likelihood = 0.0

            for seq in sequences:
                T = len(seq)

                # Forward-backward
                alpha, scale = self._forward(seq)
                beta = self._backward(seq, scale)

                # Log-likelihood
                log_likelihood += np.sum(np.log(scale + 1e-15))

                # Gamma (posterior over states)
                gamma = alpha * beta  # (T x K)
                gamma /= gamma.sum(axis=1, keepdims=True)

                # Xi (posterior over transitions)
                xi = np.zeros((T-1, self.K, self.K))
                for t in range(T-1):
                    denom = np.dot(alpha[t], self.A) * self.B[:, seq[t+1]]
                    denom = np.sum(denom)
                    for i in range(self.K):
                        xi[t, i] = alpha[t, i] * self.A[i] * self.B[:, seq[t+1]] * beta[t+1]
                    xi[t] /= xi[t].sum()

                # Accumulate Expected Counts
                pi_acc += gamma[0]

                for t in range(T-1):
                    A_acc += xi[t]

                for t in range(T):
                    B_acc[:, seq[t]] += gamma[t]

            # Normalize to re-estimate parameters
            self.pi = pi_acc / pi_acc.sum()
            self.A = A_acc / A_acc.sum(axis=1, keepdims=True)
            self.B = B_acc / B_acc.sum(axis=1, keepdims=True)

            print(f"[EM iter {it+1}/{max_iters}] log_likelihood = {log_likelihood:.4f}")
