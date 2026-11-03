import numpy as np

from .text import tokenize


class Lsa:
    """TF-IDF -> truncated SVD. Gives us a 'semantic' signal BM25 can't:
    docs that never use the query word but share its neighbourhood."""

    def __init__(self, dims=64):
        self.dims = dims

    def fit(self, texts):
        docs = [tokenize(t) for t in texts]
        vocab = sorted({w for d in docs for w in d})
        self.vocab = {w: i for i, w in enumerate(vocab)}
        X = np.zeros((len(docs), len(vocab)))
        for i, d in enumerate(docs):
            for w in d:
                X[i, self.vocab[w]] += 1
        df = (X > 0).sum(0)
        self.idf = np.log((1 + len(docs)) / (1 + df)) + 1
        X = np.log1p(X) * self.idf
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        k = min(self.dims, len(S))
        self.Vt = Vt[:k]
        self.doc_vecs = self._norm(U[:, :k] * S[:k])
        return self

    @staticmethod
    def _norm(M):
        n = np.linalg.norm(M, axis=-1, keepdims=True)
        return M / np.where(n == 0, 1, n)

    def embed(self, text):
        v = np.zeros(len(self.vocab))
        for w in tokenize(text):
            if w in self.vocab:
                v[self.vocab[w]] += 1
        v = np.log1p(v) * self.idf
        return self._norm(v @ self.Vt.T)

    def similarities(self, text):
        return self.doc_vecs @ self.embed(text)
