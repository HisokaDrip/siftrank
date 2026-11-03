from siftrank import Engine
from siftrank.engine import rrf


def test_rrf_prefers_agreement():
    assert rrf([[1, 2, 3], [2, 1, 3]])[0] in (1, 2)
    assert rrf([[1, 2], [2, 3]])[0] == 2


def test_end_to_end_on_sample_docs():
    eng = Engine(dims=16).add_dir("sample_docs").build()
    src, _ = eng.search("how does the replay buffer work", k=1)[0]
    assert "dqn" in src


def test_best_sentence():
    s = Engine.best_sentence("bloom filter", "Writes go to the WAL. A bloom filter skips tables. Done.")
    assert s.startswith("A bloom filter")
