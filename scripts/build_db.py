"""Load the Olist Kaggle CSVs into a SQLite database.

Usage:
    python scripts/build_db.py --csv-dir data/olist --db data/olist.db
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from text2sql.olist_schema import COLUMN_RENAMES, TABLES  # noqa: E402


def create_table_sql(name: str, spec: dict) -> str:
    cols = [f'    "{c}" {t}' for c, t, _ in spec["columns"]]
    if spec["primary_key"]:
        cols.append(f"    PRIMARY KEY ({', '.join(spec['primary_key'])})")
    for col, ref_table, ref_col in spec["foreign_keys"]:
        cols.append(f"    FOREIGN KEY ({col}) REFERENCES {ref_table}({ref_col})")
    return f'CREATE TABLE "{name}" (\n' + ",\n".join(cols) + "\n);"


def build(csv_dir: Path, db_path: Path) -> None:
    missing = [s["csv"] for s in TABLES.values() if not (csv_dir / s["csv"]).exists()]
    if missing:
        sys.exit(f"Missing CSV files in {csv_dir}:\n  " + "\n  ".join(missing))

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(db_path)
    try:
        for name, spec in TABLES.items():
            # utf-8-sig strips the BOM present in some copies of the translation CSV.
            # Zip codes are read as text so leading zeros are kept.
            df = pd.read_csv(csv_dir / spec["csv"], encoding="utf-8-sig",
                             dtype={c: str for c, t, _ in spec["columns"] if t == "TEXT"})
            df = df.rename(columns=COLUMN_RENAMES.get(name, {}))
            expected = [c for c, _, _ in spec["columns"]]
            extra = set(df.columns) - set(expected)
            absent = set(expected) - set(df.columns)
            if absent:
                sys.exit(f"{spec['csv']}: missing columns {sorted(absent)}")
            df = df[expected]

            if spec["primary_key"]:
                before = len(df)
                df = df.drop_duplicates(subset=spec["primary_key"])
                if len(df) < before:
                    print(f"  {name}: dropped {before - len(df)} duplicate key rows")

            con.execute(create_table_sql(name, spec))
            df.to_sql(name, con, if_exists="append", index=False)
            print(f"  {name:22s} {len(df):>8,} rows" + (f"  (ignored columns: {sorted(extra)})" if extra else ""))

        # Indexes on join / filter columns keep LLM-written queries fast.
        for name, spec in TABLES.items():
            for col, _, _ in spec["foreign_keys"]:
                con.execute(f'CREATE INDEX "idx_{name}_{col}" ON "{name}"("{col}")')
        con.execute("CREATE INDEX idx_orders_purchase ON orders(order_purchase_timestamp)")
        con.execute("CREATE INDEX idx_customers_unique ON customers(customer_unique_id)")
        con.commit()
        con.execute("ANALYZE")
    finally:
        con.close()
    print(f"Built {db_path} ({db_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv-dir", default="data/olist", type=Path)
    p.add_argument("--db", default="data/olist.db", type=Path)
    a = p.parse_args()
    build(a.csv_dir, a.db)
