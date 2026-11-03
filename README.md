# siftrank

A small local search engine for a folder of notes. It does hybrid retrieval with no models to download and no API keys:

- **BM25** over an inverted index I wrote by hand, for exact keyword hits
- **LSA**, which is TF-IDF followed by a truncated SVD in numpy. It catches chunks that are on topic but don't use your exact words
- **Reciprocal rank fusion** to merge the two rankings. Their scores aren't on the same scale, so I fuse by rank instead of trying to calibrate them
- then it picks the best-matching sentence from each hit, so you get a quick answer rather than just a filename

Docs are split into overlapping 80-word chunks, so one long file can't take over the results.

## run it

```
pip install -r requirements.txt
python -m siftrank sample_docs "why use a target network" -k 2
python -m siftrank ~/notes          # interactive, empty line quits
```

Here's a real run on the bundled sample docs:

```
indexed 5 chunks in 4 ms

1. sample_docs/dqn_notes.md
   > A target network is copied every few thousand steps so the bootstrap target doesn't chase itself.

2. sample_docs/lsm_tree.md
   > # LSM tree Writes go to an in-memory memtable and a write-ahead log for crash recovery.
```

## tests

```
python -m pytest -q
```

## what I'd do next

- swap LSA for a small sentence-embedding model when I don't mind the download
- persist the index so big folders don't get reindexed on every query
- the stemmer is about 12 lines and you can tell. A proper Porter stemmer would help
