import re
from pathlib import Path

import numpy as np

from .index import Bm25Index
from .lsa import Lsa
from .text import chunk, tokenize


def rrf(rankings, k=60):
    # reciprocal rank fusion: no score calibration needed between bm25 and cosine
    fused = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            fused[doc] = fused.get(doc, 0) + 1 / (k + rank + 1)
    return sorted(fused, key=lambda d: -fused[d])


class Engine:
    def __init__(self, dims=64):
        self.bm25 = Bm25Index()
        self.lsa = Lsa(dims)
        self.chunks = []  # (source, text)

    def add_text(self, source, text):
        for c in chunk(text):
            self.chunks.append((source, c))
            self.bm25.add(c)

    def add_dir(self, path, exts=(".md", ".txt", ".py", ".rst")):
        for p in sorted(Path(path).rglob("*")):
            if p.suffix in exts and p.is_file():
                self.add_text(str(p), p.read_text(errors="ignore"))
        return self

    def build(self):
        self.lsa.fit([c for _, c in self.chunks])
        return self

    def search(self, query, k=5, depth=50):
        lexical = [d for d, _ in self.bm25.search(query, depth)]
        semantic = list(np.argsort(-self.lsa.similarities(query))[:depth])
        return [(self.chunks[d][0], self.chunks[d][1]) for d in rrf([lexical, semantic])[:k]]

    @staticmethod
    def best_sentence(query, text):
        """Extractive 'answer': the sentence with the most query-term overlap."""
        q = set(tokenize(query))
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        return max(sents, key=lambda s: len(q & set(tokenize(s))), default="")
