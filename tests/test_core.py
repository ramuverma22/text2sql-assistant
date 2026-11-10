import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pandas as pd
import pytest

from conftest import OracleLLM, ScriptedLLM
from eval.compare import results_match
from text2sql.agent import AgentConfig, Text2SQLAgent, extract_sql
from text2sql.charts import choose_chart
from text2sql.db import Database, UnsafeSQLError, validate_sql
from text2sql.llm import ChatLLM
from text2sql.schema import SchemaIndex, connect_tables


# ---------------- safety ----------------
@pytest.mark.parametrize("sql", [
    "SELECT 1",
    "select count(*) from orders;",
    "WITH x AS (SELECT 1 AS a) SELECT a FROM x",
    "SELECT replace(customer_city, 'a', 'b') FROM customers",
    "SELECT 'drop table orders' AS txt",
    "SELECT order_id -- delete everything\nFROM orders",
])
def test_allowed(sql):
    validate_sql(sql)


@pytest.mark.parametrize("sql", [
    "DROP TABLE orders",
    "SELECT 1; DROP TABLE orders",
    "WITH x AS (SELECT 1) DELETE FROM orders",
    "PRAGMA table_info(orders)",
    "ATTACH DATABASE 'x.db' AS x",
    "UPDATE orders SET order_status = 'x'",
    "WITH x AS (SELECT 1) REPLACE INTO orders SELECT * FROM orders",
    "",
])
def test_blocked(sql):
    with pytest.raises(UnsafeSQLError):
        validate_sql(sql)


def test_connection_is_read_only(demo_db):
    con = Database(demo_db).connect()
    with pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM orders")
    con.close()


def test_row_cap(demo_db):
    r = Database(demo_db, max_rows=10).run("SELECT * FROM orders")
    assert len(r.df) == 10 and r.truncated


def test_timeout(demo_db):
    db = Database(demo_db, timeout_s=0.2)
    slow = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
    with pytest.raises(TimeoutError):
        db.run(slow)


# ---------------- comparison ----------------
def test_compare_rules():
    gold = [("SP", 10.123), ("RJ", 5)]
    assert results_match(gold, [("RJ", 5.0), ("SP", 10.1249)])            # order, rounding, int/float
    assert results_match(gold, [(10.12, "sp"), (5, "rj")])                 # column order, case
    assert results_match(gold, [("SP", 99, 10.12), ("RJ", 98, 5)])         # extra column allowed
    assert results_match([("SP",)], [("SP", 1234)])                        # gold label only
    assert results_match([(14,)], [("14",)])                               # numeric string
    assert not results_match(gold, [("SP", 10.2), ("RJ", 5)])              # wrong value
    assert not results_match(gold, [("SP", 10.12)])                        # missing row
    assert not results_match(gold, [("SP",), ("RJ",)])                     # missing column
    assert results_match([], [])


# ---------------- schema retrieval ----------------
def test_bridge_tables():
    t = connect_tables(["products", "customers"])
    assert {"order_items", "orders"} <= set(t)


@pytest.mark.parametrize("q,needed", [
    ("Average review score by customer state", {"reviews", "customers", "orders"}),
    ("Revenue by product category in English", {"order_items", "products", "category_translation"}),
    ("Which payment type is most common?", {"payments"}),
    ("How many sellers are in SP?", {"sellers"}),
])
def test_retrieval(demo_db, q, needed):
    assert needed <= set(SchemaIndex(demo_db).retrieve(q))


def test_value_hints(demo_db):
    text = SchemaIndex(demo_db).describe(["payments", "orders"])
    assert "credit_card" in text and "delivered" in text and "JOINS" in text


# ---------------- agent ----------------
def test_extract_sql():
    assert extract_sql("Here:\n```sql\nSELECT 1\n```") == "SELECT 1"
    assert extract_sql("```\nSELECT 2\n```") == "SELECT 2"
    assert extract_sql("Sure! SELECT 3") == "SELECT 3"


def test_self_correction(demo_db):
    llm = ScriptedLLM(["```sql\nSELECT COUNT(*) FROM order\n```", "```sql\nSELECT COUNT(*) FROM orders\n```"])
    res = Text2SQLAgent(demo_db, llm, AgentConfig()).ask("How many orders?")
    assert res.ok and len(res.attempts) == 2 and res.attempts[0].error
    assert "syntax error" in llm.calls[1][-1]["content"] or "no such table" in llm.calls[1][-1]["content"]
    assert res.df.iloc[0, 0] == 1500


def test_no_self_correction_reports_error(demo_db):
    llm = ScriptedLLM(["```sql\nSELECT nope FROM orders\n```"])
    res = Text2SQLAgent(demo_db, llm, AgentConfig.baseline()).ask("x")
    assert not res.ok and "no such column" in res.error


def test_unsafe_sql_is_never_run(demo_db):
    llm = ScriptedLLM(["```sql\nDELETE FROM orders\n```"] * 3)
    res = Text2SQLAgent(demo_db, llm, AgentConfig()).ask("delete everything")
    assert not res.ok and "UnsafeSQLError" in res.error
    assert Database(demo_db).run("SELECT COUNT(*) FROM orders").df.iloc[0, 0] == 1500


def test_empty_result_check(demo_db):
    llm = ScriptedLLM([
        "```sql\nSELECT COUNT(*) AS n FROM orders WHERE order_status = 'Delivered' GROUP BY order_status\n```",
        "```sql\nSELECT COUNT(*) AS n FROM orders WHERE order_status = 'delivered' GROUP BY order_status\n```",
    ])
    res = Text2SQLAgent(demo_db, llm, AgentConfig()).ask("delivered orders?")
    assert res.ok and len(res.attempts) == 2 and res.df.iloc[0, 0] > 0


def test_prompt_contents(demo_db):
    llm = ScriptedLLM(["```sql\nSELECT 1\n```"])
    Text2SQLAgent(demo_db, llm, AgentConfig(schema_mode="rag")).ask("What percentage of orders were paid by voucher?")
    prompt = llm.calls[0][1]["content"]
    assert "TABLE payments" in prompt and "Similar solved examples" in prompt and "TABLE sellers" not in prompt


def test_summary(demo_db):
    llm = ScriptedLLM(["```sql\nSELECT order_status, COUNT(*) AS n FROM orders GROUP BY 1\n```", "Most orders were delivered."])
    res = Text2SQLAgent(demo_db, llm).ask("orders by status", summarize=True)
    assert res.answer == "Most orders were delivered." and res.chart["kind"] == "bar"


# ---------------- charts ----------------
def test_charts():
    assert choose_chart(pd.DataFrame({"month": ["2017-01", "2017-02"], "rev": [1.0, 2.0]}))["kind"] == "line"
    assert choose_chart(pd.DataFrame({"state": ["SP", "RJ"], "n": [3, 2]}))["kind"] == "bar"
    assert choose_chart(pd.DataFrame({"n": [3]})) is None


# ---------------- LLM client against a fake OpenAI-compatible server ----------------
def test_llm_client_with_retry():
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, self.headers["Authorization"], body, self.headers["User-Agent"]))
            if len(seen) == 1:
                self.send_response(429)
                self.send_header("Retry-After", "0")
                self.end_headers()
                return
            out = json.dumps({"choices": [{"message": {"content": "```sql\nSELECT 1\n```"}}],
                              "usage": {"prompt_tokens": 12, "completion_tokens": 3}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        llm = ChatLLM(model="m", base_url=f"http://127.0.0.1:{srv.server_port}/v1", api_key="k")
        r = llm.chat([{"role": "user", "content": "hi"}])
    finally:
        srv.shutdown()
    assert r.text.startswith("```sql") and r.prompt_tokens == 12
    assert len(seen) == 2 and seen[1][0] == "/v1/chat/completions" and seen[1][1] == "Bearer k"
    assert seen[1][2]["model"] == "m" and seen[1][2]["temperature"] == 0.0
    # urllib's default "Python-urllib" agent is blocked by Cloudflare (Groq: 403, error 1010).
    assert not seen[1][3].startswith("Python-urllib")


# ---------------- end-to-end evaluation ----------------
def test_eval_pipeline_oracle(demo_db, questions, tmp_path):
    from eval.run_eval import run
    llm = OracleLLM(questions, break_first_ids={1, 20, 43})
    args = SimpleNamespace(questions=str(tmp_path / "q.json"), limit=None, db=demo_db, sleep=0,
                           out=str(tmp_path / "res"), configs=["baseline", "full_rag"])
    (tmp_path / "q.json").write_text(json.dumps(questions))
    run(args, llm)
    summary = {s["config"]: s for s in json.loads((tmp_path / "res" / "summary.json").read_text())}
    n = len(questions)
    # Without self-correction the 3 broken queries fail; with it everything is fixed.
    assert summary["baseline"]["accuracy"] == pytest.approx(100 * (n - 3) / n)
    assert summary["full_rag"]["accuracy"] == 100.0
    assert summary["full_rag"]["fixed_by_retry"] == 3
