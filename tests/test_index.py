from siftrank import Bm25Index


def build():
    ix = Bm25Index()
    for t in ["port scanning with nmap", "baking sourdough bread", "detecting port scans in pcap files"]:
        ix.add(t)
    return ix


def test_relevant_doc_ranks_first():
    top = build().search("port scan detection")
    assert top[0][0] == 2


def test_unknown_terms_give_nothing():
    assert build().search("quantum") == []


def test_rare_term_beats_common():
    ix = build()
    assert ix.idf("sourdough") > ix.idf("port")
