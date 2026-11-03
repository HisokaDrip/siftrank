import math
from collections import Counter, defaultdict

from .text import tokenize


class Bm25Index:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.postings = defaultdict(dict)  # term -> {doc_id: tf}
        self.lengths = []

    def add(self, text):
        doc_id = len(self.lengths)
        toks = tokenize(text)
        self.lengths.append(len(toks))
        for term, tf in Counter(toks).items():
            self.postings[term][doc_id] = tf
        return doc_id

    def idf(self, term):
        n, df = len(self.lengths), len(self.postings.get(term, ()))
        # the +1 inside the log keeps idf positive for very common terms (Lucene does the same)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def scores(self, query):
        if not self.lengths:
            return {}
        avg = sum(self.lengths) / len(self.lengths)
        out = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf(term)
            for d, tf in self.postings.get(term, {}).items():
                norm = self.k1 * (1 - self.b + self.b * self.lengths[d] / avg)
                out[d] += idf * tf * (self.k1 + 1) / (tf + norm)
        return dict(out)

    def search(self, query, k=5):
        s = self.scores(query)
        return sorted(s.items(), key=lambda x: -x[1])[:k]
