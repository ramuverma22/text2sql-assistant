"""Schema context for the LLM.

Two modes:
* `raw_schema()`  - plain CREATE TABLE statements (used as the baseline).
* `SchemaIndex`   - documented schema with example values, retrieved per
  question with BM25 ("schema RAG"), plus any bridge tables needed to join
  the retrieved tables (found via the foreign-key graph).
"""
import math
import re
import sqlite3
from collections import Counter, deque

from .olist_schema import GLOSSARY, TABLES

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "in", "on", "for", "to", "by", "and", "or", "is", "are",
    "what", "which", "how", "many", "much", "per", "with", "each", "from", "show",
    "me", "list", "give", "top", "most", "highest", "lowest", "average", "total",
    "number", "count", "did", "does", "do", "was", "were", "has", "have", "that",
    "this", "it", "its", "their", "be", "as", "at", "than", "id", "ids",
}


def tokenize(text: str) -> list[str]:
    out = []
    for t in _TOKEN.findall(text.lower().replace("_", " ")):
        if t in _STOP:
            continue
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]  # crude plural stemming: orders -> order
        out.append(t)
    return out


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.docs)) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        res = []
        for tf, d in zip(self.tf, self.docs):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            res.append(s)
        return res


def raw_schema(db_path: str) -> str:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name").fetchall()
    finally:
        con.close()
    return "\n\n".join(r[0] for r in rows)


def _fk_graph() -> dict[str, set[str]]:
    g = {t: set() for t in TABLES}
    for t, spec in TABLES.items():
        for _, ref, _ in spec["foreign_keys"]:
            g[t].add(ref)
            g[ref].add(t)
    return g


def _shortest_path(g, a, b):
    prev, q = {a: None}, deque([a])
    while q:
        n = q.popleft()
        if n == b:
            path = []
            while n is not None:
                path.append(n)
                n = prev[n]
            return path[::-1]
        for m in g[n]:
            if m not in prev:
                prev[m] = n
                q.append(m)
    return [a, b]


def connect_tables(selected: list[str]) -> list[str]:
    """Add the bridge tables needed so every selected table can be joined.
    E.g. {products, customers} -> also needs order_items and orders."""
    g = _fk_graph()
    result = list(dict.fromkeys(selected))
    if len(result) < 2:
        return result
    anchor = result[0]
    for t in list(result[1:]):
        for n in _shortest_path(g, anchor, t):
            if n not in result:
                result.append(n)
    return result


class SchemaIndex:
    def __init__(self, db_path: str, max_distinct: int = 30):
        self.db_path = db_path
        self.tables = list(TABLES)
        self.value_hints = self._load_value_hints(max_distinct)
        self.bm25 = BM25([self._search_text(t) for t in self.tables])

    def _load_value_hints(self, max_distinct: int) -> dict[tuple, list[str]]:
        """Distinct values of low-cardinality text columns, e.g. order_status.
        These stop the model guessing 'Credit Card' when the data says 'credit_card'."""
        overrides = {("category_translation", "product_category_name_english"): 80}
        skip = {("products", "product_category_name"), ("category_translation", "product_category_name")}
        hints = {}
        con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            for t, spec in TABLES.items():
                for col, typ, _ in spec["columns"]:
                    if typ != "TEXT" or (t, col) in skip or col.endswith("_id") or "date" in col or "timestamp" in col:
                        continue
                    limit = overrides.get((t, col), max_distinct)
                    vals = con.execute(
                        f'SELECT DISTINCT "{col}" FROM "{t}" WHERE "{col}" IS NOT NULL ORDER BY 1 LIMIT {limit + 1}'
                    ).fetchall()
                    if 0 < len(vals) <= limit:
                        hints[(t, col)] = [v[0] for v in vals]
        except sqlite3.OperationalError:
            pass  # table missing: hints are optional
        finally:
            con.close()
        return hints

    def _search_text(self, t: str) -> str:
        spec = TABLES[t]
        cols = " ".join(f"{c} {d}" for c, _, d in spec["columns"])
        return f"{t} {spec['description']} {spec['keywords']} {spec['keywords']} {cols}"

    def retrieve(self, question: str, k: int = 4) -> list[str]:
        scores = self.bm25.scores(question)
        ranked = sorted(zip(self.tables, scores), key=lambda x: -x[1])
        picked = [t for t, s in ranked[:k] if s > 0]
        if not picked:
            return list(self.tables)
        # A year like "2017" never matches a keyword, but order dates live in `orders`.
        if re.search(r"\b20\d{2}\b", question) and "orders" not in picked:
            picked.append("orders")
        return connect_tables(picked)

    def describe(self, tables: list[str] | None = None) -> str:
        tables = tables or self.tables
        parts = []
        for t in tables:
            spec = TABLES[t]
            lines = [f"TABLE {t}  -- {spec['description']}"]
            for col, typ, desc in spec["columns"]:
                line = f"  {col} {typ}  -- {desc}"
                vals = self.value_hints.get((t, col))
                if vals:
                    line += f". Values: {', '.join(map(str, vals))}"
                lines.append(line)
            if spec["primary_key"]:
                lines.append(f"  PRIMARY KEY ({', '.join(spec['primary_key'])})")
            parts.append("\n".join(lines))
        joins = sorted({
            f"{t}.{c} = {ref}.{rc}"
            for t in tables for c, ref, rc in TABLES[t]["foreign_keys"] if ref in tables
        })
        text = "\n\n".join(parts)
        if joins:
            text += "\n\nJOINS:\n" + "\n".join(f"  {j}" for j in joins)
        text += "\n\nBUSINESS RULES:\n" + "\n".join(f"  - {g}" for g in GLOSSARY)
        return text


class ExampleIndex:
    """Retrieves the most similar solved examples (question -> SQL) for few-shot prompting."""

    def __init__(self, examples: list[dict]):
        self.examples = examples
        self.bm25 = BM25([e["question"] for e in examples]) if examples else None

    def retrieve(self, question: str, k: int = 3) -> list[dict]:
        if not self.bm25:
            return []
        scores = self.bm25.scores(question)
        ranked = sorted(zip(self.examples, scores), key=lambda x: -x[1])
        return [e for e, s in ranked[:k] if s > 0]
