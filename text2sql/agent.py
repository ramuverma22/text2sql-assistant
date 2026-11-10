"""The text-to-SQL pipeline:

question -> schema context (raw / documented / retrieved) -> similar examples
-> LLM writes SQL -> safety check + run -> on error, feed the error back
and retry (self-correction) -> optional plain-English answer + chart.
"""
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .charts import choose_chart
from .db import Database, UnsafeSQLError
from .prompts import (EMPTY_RESULT_PROMPT, FIX_ERROR_PROMPT, SUMMARY_PROMPT,
                      SYSTEM_PROMPT, build_user_prompt)
from .schema import ExampleIndex, SchemaIndex, raw_schema

EXAMPLES_PATH = Path(__file__).with_name("examples.json")


@dataclass
class AgentConfig:
    schema_mode: str = "rag"      # "raw" (CREATE TABLEs) | "docs" (all tables, documented) | "rag" (retrieved tables)
    use_fewshot: bool = True      # add the most similar solved examples to the prompt
    self_correct: bool = True     # retry with the error message when a query fails
    max_fix_attempts: int = 2
    check_empty: bool = True      # ask the model to double-check a 0-row result once
    rag_k: int = 4
    fewshot_k: int = 3

    @classmethod
    def baseline(cls):
        return cls(schema_mode="raw", use_fewshot=False, self_correct=False, check_empty=False)


@dataclass
class Attempt:
    sql: str
    error: str | None = None
    rows: int | None = None


@dataclass
class AgentResult:
    question: str
    sql: str | None = None
    df: pd.DataFrame | None = None
    error: str | None = None
    truncated: bool = False
    attempts: list[Attempt] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    examples_used: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    answer: str | None = None
    chart: dict | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.df is not None


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model reply (code block preferred)."""
    blocks = re.findall(r"```(?:sql|sqlite)?\s*\n?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[-1].strip()
    m = re.search(r"\b(with|select)\b.*", text, flags=re.DOTALL | re.IGNORECASE)
    return (m.group(0) if m else text).strip()


class Text2SQLAgent:
    def __init__(self, db_path: str, llm, config: AgentConfig | None = None,
                 examples_path: Path | str | None = EXAMPLES_PATH):
        self.db = Database(db_path)
        self.llm = llm
        self.config = config or AgentConfig()
        self.schema_index = SchemaIndex(db_path)
        self._raw_schema = raw_schema(db_path)
        examples = []
        if examples_path and Path(examples_path).exists():
            examples = json.loads(Path(examples_path).read_text())
        self.example_index = ExampleIndex(examples)

    # ---------- context ----------
    def schema_context(self, question: str) -> tuple[str, list[str]]:
        mode = self.config.schema_mode
        if mode == "raw":
            return self._raw_schema, self.db.table_names()
        if mode == "docs":
            tables = list(self.schema_index.tables)
        elif mode == "rag":
            tables = self.schema_index.retrieve(question, k=self.config.rag_k)
        else:
            raise ValueError(f"Unknown schema_mode {mode!r}")
        return self.schema_index.describe(tables), tables

    # ---------- main loop ----------
    def ask(self, question: str, summarize: bool = False) -> AgentResult:
        cfg = self.config
        res = AgentResult(question=question)
        schema_text, res.tables = self.schema_context(question)
        examples = self.example_index.retrieve(question, cfg.fewshot_k) if cfg.use_fewshot else []
        res.examples_used = [e["question"] for e in examples]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, schema_text, examples)},
        ]
        fixes_left = cfg.max_fix_attempts if cfg.self_correct else 0
        empty_checked = not cfg.check_empty

        while True:
            reply = self.llm.chat(messages)
            res.prompt_tokens += reply.prompt_tokens
            res.completion_tokens += reply.completion_tokens
            sql = extract_sql(reply.text)
            attempt = Attempt(sql=sql)
            res.attempts.append(attempt)
            try:
                qr = self.db.run(sql)
            except (UnsafeSQLError, sqlite3.Error, TimeoutError) as e:
                attempt.error = f"{type(e).__name__}: {e}"
                if fixes_left > 0:
                    fixes_left -= 1
                    messages += [{"role": "assistant", "content": reply.text},
                                 {"role": "user", "content": FIX_ERROR_PROMPT.format(error=attempt.error)}]
                    continue
                res.sql, res.error = sql, attempt.error
                break

            attempt.rows = len(qr.df)
            if qr.df.empty and not empty_checked:
                empty_checked = True
                messages += [{"role": "assistant", "content": reply.text},
                             {"role": "user", "content": EMPTY_RESULT_PROMPT}]
                continue
            res.sql, res.df, res.truncated = sql, qr.df, qr.truncated
            break

        if res.ok:
            res.chart = choose_chart(res.df)
            if summarize:
                res.answer = self.summarize(res)
        return res

    def summarize(self, res: AgentResult) -> str:
        df = res.df
        # Format floats as 1,010,270.37 rather than the default 1.01027e+06,
        # which the model would otherwise copy into its answer.
        if _has_tabulate():
            table = df.head(15).to_markdown(index=False, floatfmt=",.2f")
        else:
            table = df.head(15).to_string(index=False, float_format=lambda v: f"{v:,.2f}")
        prompt = SUMMARY_PROMPT.format(
            question=res.question, sql=res.sql, n_rows=len(df),
            truncated_note=", truncated" if res.truncated else "", table=table)
        # Leave headroom: reasoning models spend part of this budget thinking
        # before they write the answer.
        reply = self.llm.chat([{"role": "user", "content": prompt}], max_tokens=1000)
        res.prompt_tokens += reply.prompt_tokens
        res.completion_tokens += reply.completion_tokens
        return reply.text.strip()


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False
