import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_db  # noqa: E402
import make_demo_data  # noqa: E402

from text2sql.llm import LLMResponse  # noqa: E402


@pytest.fixture(scope="session")
def demo_db(tmp_path_factory):
    d = tmp_path_factory.mktemp("demo")
    make_demo_data.main(d / "csv", n_orders=1500, seed=1)
    db = d / "demo.db"
    build_db.build(d / "csv", db)
    return str(db)


@pytest.fixture(scope="session")
def questions():
    return json.loads((ROOT / "eval" / "questions.json").read_text())


class ScriptedLLM:
    """Returns pre-written replies in order and records the prompts it saw."""

    model = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, max_tokens=800):
        self.calls.append(messages)
        return LLMResponse(self.replies.pop(0), prompt_tokens=100, completion_tokens=20)


class OracleLLM:
    """Answers each benchmark question with its gold SQL (optionally broken on the
    first try) - lets us test the whole evaluation loop without an API."""

    model = "oracle"

    def __init__(self, questions, break_first_ids=()):
        self.gold = {q["question"]: q["sql"] for q in questions}
        self.break_first = set(break_first_ids)
        self.ids = {q["question"]: q["id"] for q in questions}

    def chat(self, messages, max_tokens=800):
        first = messages[1]["content"]
        q = first.rsplit("### Question\n", 1)[-1].strip()
        sql = self.gold[q]
        if self.ids[q] in self.break_first and len(messages) == 2:
            sql = sql.replace("FROM", "FRM", 1)  # syntax error on first attempt
        return LLMResponse(f"```sql\n{sql}\n```", 50, 10)
