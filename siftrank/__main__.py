import argparse
import time

from .engine import Engine


def main():
    ap = argparse.ArgumentParser(prog="siftrank", description="hybrid bm25 + lsa search over a folder")
    ap.add_argument("folder")
    ap.add_argument("query", nargs="?")
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()

    t0 = time.perf_counter()
    eng = Engine().add_dir(args.folder).build()
    print(f"indexed {len(eng.chunks)} chunks in {(time.perf_counter() - t0) * 1000:.0f} ms\n")

    queries = [args.query] if args.query else iter(lambda: input("? "), "")
    for q in queries:
        for i, (src, text) in enumerate(eng.search(q, args.k), 1):
            print(f"{i}. {src}\n   > {eng.best_sentence(q, text)}\n")


if __name__ == "__main__":
    main()
