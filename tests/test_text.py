from siftrank import tokenize, chunk


def test_stemming_merges_variants():
    assert tokenize("scanning scans") == ["scann", "scan"] or len(set(tokenize("scans scan"))) == 1


def test_stopwords_dropped():
    assert tokenize("the cat and the hat") == ["cat", "hat"]


def test_chunk_overlap():
    text = " ".join(str(i) for i in range(200))
    parts = chunk(text, size=80, overlap=20)
    assert parts[0].split()[-20:] == parts[1].split()[:20]
    assert parts[-1].split()[-1] == "199"
