"""Execution-accuracy comparison: does the predicted query return the same
answer as the gold query?

Rules (documented in the README so the reported number is honest):
* Row order is ignored (results are compared as multisets).
* Column names and column order are ignored.
* Extra columns in the prediction are allowed, as long as some set of its
  columns reproduces the gold result exactly.
* Numbers are rounded to 2 decimals; whole-number floats equal ints;
  numeric strings ('2017', '16') equal numbers; text is compared case-insensitively.
"""
from collections import Counter

_NUM_PLACES = 2


def norm(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        try:
            v = float(s)
        except ValueError:
            return s.lower()
    if isinstance(v, (int, float)):
        f = round(float(v), _NUM_PLACES)
        if f == 0:
            f = 0.0  # -0.0 -> 0.0
        return int(f) if f.is_integer() else f
    return str(v).lower()


def _normalize(rows):
    return [tuple(norm(v) for v in r) for r in rows]


def results_match(gold_rows, pred_rows) -> bool:
    gold, pred = _normalize(gold_rows), _normalize(pred_rows)
    if len(gold) != len(pred):
        return False
    if not gold:
        return True
    k, m = len(gold[0]), len(pred[0])
    if m < k:
        return False

    gold_counter = Counter(gold)
    gold_cols = [sorted(map(repr, col)) for col in zip(*gold)]
    pred_cols = [sorted(map(repr, col)) for col in zip(*pred)]
    # Candidate predicted columns for each gold column: same multiset of values.
    candidates = [[j for j in range(m) if pred_cols[j] == gc] for gc in gold_cols]
    if any(not c for c in candidates):
        return False

    def search(i, used, mapping):
        if i == k:
            return Counter(tuple(r[j] for j in mapping) for r in pred) == gold_counter
        for j in candidates[i]:
            if j not in used:
                if search(i + 1, used | {j}, mapping + [j]):
                    return True
        return False

    return search(0, frozenset(), [])
