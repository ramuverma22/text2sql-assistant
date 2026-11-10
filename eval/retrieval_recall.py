"""Check schema retrieval on its own (no LLM needed): for each benchmark
question, are all tables used by the gold SQL among the retrieved tables?

Usage:  python eval/retrieval_recall.py --db data/olist.db
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from text2sql.olist_schema import TABLES  # noqa: E402
from text2sql.schema import SchemaIndex  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--db", default="data/olist.db")
a = p.parse_args()

idx = SchemaIndex(a.db)
qs = json.loads((ROOT / "eval" / "questions.json").read_text())
hits, sizes = 0, []
for q in qs:
    need = {t for t in TABLES if re.search(rf"\b{t}\b", q["sql"])}
    got = idx.retrieve(q["question"])
    sizes.append(len(got))
    if need <= set(got):
        hits += 1
    else:
        print(f"#{q['id']}: missing {sorted(need - set(got))} - {q['question']}")
print(f"Retrieval recall: {hits}/{len(qs)} questions got every table they need "
      f"({100 * hits / len(qs):.0f}%); average {sum(sizes) / len(sizes):.1f} of {len(TABLES)} tables sent.")
