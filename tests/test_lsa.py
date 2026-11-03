import numpy as np
from siftrank import Lsa

DOCS = [
    "neural network gradient descent training loss",
    "gradient descent optimizer learning rate",
    "firewall rules block inbound traffic",
    "firewall logs show blocked packets traffic",
]


def test_topics_separate():
    lsa = Lsa(dims=2).fit(DOCS)
    sims = lsa.similarities("training loss")
    assert np.argmax(sims) in (0, 1)
    assert sims[1] > sims[3]  # doc 1 never says "training" but shares the topic


def test_unit_vectors():
    lsa = Lsa(dims=3).fit(DOCS)
    assert np.allclose(np.linalg.norm(lsa.doc_vecs, axis=1), 1)
