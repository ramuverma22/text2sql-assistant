"""Read-only SQL execution with three layers of protection:

1. `validate_sql` rejects anything that is not a single SELECT / WITH query.
2. The database is opened in SQLite read-only mode (`mode=ro`) with `query_only`,
   so even a query that slipped past validation cannot modify data.
3. Queries are stopped after a time limit and results are capped at a row limit.
"""
import re
import sqlite3
import time
from dataclasses import dataclass, field

import pandas as pd

FORBIDDEN = re.compile(
    # REPLACE alone is allowed because replace() is a normal string function.
    r"\b(insert|update|delete|drop|alter|create|replace\s+into|attach|detach|pragma|vacuum|"
    r"reindex|analyze|begin|commit|rollback|savepoint|release|load_extension)\b",
    re.IGNORECASE,
)


class UnsafeSQLError(ValueError):
    pass


def _strip_comments_and_strings(sql: str) -> str:
    """Remove comments and string literals so keywords inside them are not flagged."""
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"'(?:[^']|'')*'", "''", sql)
    sql = re.sub(r'"(?:[^"]|"")*"', '""', sql)
    return sql


def validate_sql(sql: str) -> str:
    """Return the cleaned query, or raise UnsafeSQLError."""
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise UnsafeSQLError("Empty query.")
    bare = _strip_comments_and_strings(sql)
    if ";" in bare:
        raise UnsafeSQLError("Only a single SQL statement is allowed.")
    if not re.match(r"^\s*(select|with)\b", bare, re.IGNORECASE):
        raise UnsafeSQLError("Only SELECT queries (optionally starting with WITH) are allowed.")
    bad = FORBIDDEN.search(bare)
    if bad:
        raise UnsafeSQLError(f"Keyword '{bad.group(0).upper()}' is not allowed (read-only access).")
    return sql


@dataclass
class QueryResult:
    df: pd.DataFrame
    truncated: bool = False
    elapsed_s: float = 0.0
    columns: list = field(default_factory=list)


class Database:
    def __init__(self, path: str, timeout_s: float = 15.0, max_rows: int = 5000):
        self.path = path
        self.timeout_s = timeout_s
        self.max_rows = max_rows

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        con.execute("PRAGMA query_only = ON")
        return con

    def run(self, sql: str) -> QueryResult:
        sql = validate_sql(sql)
        con = self.connect()
        deadline = time.monotonic() + self.timeout_s
        # Returning non-zero from the progress handler aborts the running query.
        con.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        start = time.monotonic()
        try:
            cur = con.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(self.max_rows + 1)
        except sqlite3.OperationalError as e:
            if "interrupted" in str(e).lower():
                raise TimeoutError(f"Query took longer than {self.timeout_s:.0f}s and was stopped.") from e
            raise
        finally:
            con.close()
        truncated = len(rows) > self.max_rows
        rows = rows[: self.max_rows]
        df = pd.DataFrame.from_records(rows, columns=cols)
        return QueryResult(df=df, truncated=truncated, elapsed_s=time.monotonic() - start, columns=cols)

    def raw_rows(self, sql: str) -> list[tuple]:
        """Used by the evaluator: all rows, no row cap."""
        sql = validate_sql(sql)
        con = self.connect()
        deadline = time.monotonic() + self.timeout_s
        con.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        try:
            return con.execute(sql).fetchall()
        finally:
            con.close()

    def table_names(self) -> list[str]:
        con = self.connect()
        try:
            return [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        finally:
            con.close()
