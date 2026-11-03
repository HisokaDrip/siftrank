import re

# small stoplist on purpose - big ones hurt short technical queries
STOP = set("""a an the and or of to in on for is are was were be by with as at it this that
from these those its into than then there their they we you i not but if so can will""".split())

_word = re.compile(r"[a-z0-9]+")


def stem(w):
    # crude suffix stripping, good enough to merge "scanning"/"scans"/"scan"
    for suf in ("ization", "ations", "ation", "ings", "ing", "edly", "ies", "ers", "ed", "es", "ly", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            return w + "y" if suf == "ies" else w
    return w


def tokenize(text):
    return [stem(t) for t in _word.findall(text.lower()) if t not in STOP]


def chunk(text, size=80, overlap=20):
    """Split into overlapping word windows so long docs don't drown the scores."""
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    out, step = [], size - overlap
    for i in range(0, len(words) - overlap, step):
        out.append(" ".join(words[i : i + size]))
    return out
