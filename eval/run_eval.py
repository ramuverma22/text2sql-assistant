"""Benchmark the assistant: execution accuracy on eval/questions.json.

Runs one or more pipeline configurations (an ablation) and writes:
  eval/results/<config>.jsonl   one line per question (SQL, correct?, errors, tokens)
  eval/results/summary.md       accuracy table to paste into your README

Usage:
    python eval/run_eval.py --db data/olist.db                       # all configs
    python eval/run_eval.py --db data/olist.db --configs baseline full_rag
    python eval/run_eval.py --db data/olist.db --limit 10 --sleep 2  # quick / rate-limited run
"""
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.compare import results_match  # noqa: E402
from text2sql.agent import AgentConfig, Text2SQLAgent  # noqa: E402
from text2sql.db import Database  # noqa: E402

# Each config adds one technique to the previous one, so the table shows
# how much each step contributes.
CONFIGS = {
    "baseline": AgentConfig.baseline(),
    "docs": AgentConfig(schema_mode="docs", use_fewshot=False, self_correct=False, check_empty=False),
    "docs_fewshot": AgentConfig(schema_mode="docs", use_fewshot=True, self_correct=False, check_empty=False),
    "full_docs": AgentConfig(schema_mode="docs", use_fewshot=True, self_correct=True, check_empty=True),
    "full_rag": AgentConfig(schema_mode="rag", use_fewshot=True, self_correct=True, check_empty=True),
}
LABELS = {
    "baseline": "Baseline (raw CREATE TABLE schema)",
    "docs": "+ documented schema & example values",
    "docs_fewshot": "+ retrieved few-shot examples",
    "full_docs": "+ self-correction (all tables)",
    "full_rag": "+ self-correction, schema RAG (retrieved tables only)",
}


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


def evaluate(agent: Text2SQLAgent, questions, gold, sleep_s: float, out_path: Path):
    rows = []
    with out_path.open("w") as f:
        for q in questions:
            t0 = time.monotonic()
            try:
                res = agent.ask(q["question"])
                pred_rows = None
                if res.ok:
                    pred_rows = agent.db.raw_rows(res.sql)
                correct = pred_rows is not None and results_match(gold[q["id"]], pred_rows)
                rec = {
                    "id": q["id"], "difficulty": q["difficulty"], "question": q["question"],
                    "correct": correct, "pred_sql": res.sql, "gold_sql": q["sql"], "error": res.error,
                    "n_attempts": len(res.attempts), "attempts": [asdict(a) for a in res.attempts],
                    "tables": res.tables, "prompt_tokens": res.prompt_tokens,
                    "completion_tokens": res.completion_tokens,
                }
            except Exception as e:  # API failure etc.: count as wrong, keep going
                rec = {"id": q["id"], "difficulty": q["difficulty"], "question": q["question"],
                       "correct": False, "pred_sql": None, "gold_sql": q["sql"],
                       "error": f"{type(e).__name__}: {e}", "n_attempts": 0, "attempts": [],
                       "tables": [], "prompt_tokens": 0, "completion_tokens": 0}
            rec["latency_s"] = round(time.monotonic() - t0, 2)
            rows.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()  # keep finished questions on disk if the run is interrupted
            mark = "OK " if rec["correct"] else "XX "
            print(f"  {mark} #{q['id']:>2} [{q['difficulty']:6s}] {q['question'][:70]}")
            if sleep_s:
                time.sleep(sleep_s)
    return rows


def summarize(name, rows):
    def acc(rs):
        return 100.0 * sum(r["correct"] for r in rs) / len(rs) if rs else float("nan")
    by = {d: [r for r in rows if r["difficulty"] == d] for d in ("easy", "medium", "hard")}
    fixed = sum(1 for r in rows if r["correct"] and r["n_attempts"] > 1)
    return {
        "config": name, "n": len(rows), "accuracy": acc(rows),
        **{f"acc_{d}": acc(rs) for d, rs in by.items()},
        "fixed_by_retry": fixed,
        "exec_errors": sum(1 for r in rows if r["error"]),
        "avg_prompt_tokens": sum(r["prompt_tokens"] for r in rows) / max(len(rows), 1),
        "avg_latency_s": sum(r["latency_s"] for r in rows) / max(len(rows), 1),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="data/olist.db")
    p.add_argument("--questions", default=str(ROOT / "eval" / "questions.json"))
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--limit", type=int, default=None, help="only the first N questions")
    p.add_argument("--sleep", type=float, default=0.0, help="seconds between questions (rate limits)")
    p.add_argument("--out", default=str(ROOT / "eval" / "results"))
    a = p.parse_args()

    load_env()
    from text2sql.llm import ChatLLM
    llm = ChatLLM()
    run(a, llm)


def run(a, llm):
    questions = json.loads(Path(a.questions).read_text())[: a.limit]
    db = Database(a.db, timeout_s=60)
    gold = {}
    for q in questions:
        gold[q["id"]] = db.raw_rows(q["sql"])

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name in a.configs:
        print(f"\n=== {name}: {LABELS[name]} ===")
        agent = Text2SQLAgent(a.db, llm, CONFIGS[name])
        rows = evaluate(agent, questions, gold, a.sleep, out_dir / f"{name}.jsonl")
        s = summarize(name, rows)
        summaries.append(s)
        print(f"  -> accuracy {s['accuracy']:.1f}%  ({sum(r['correct'] for r in rows)}/{len(rows)})")

    model = getattr(llm, "model", "?")
    lines = [
        f"Model: `{model}`  |  Questions: {len(questions)}  |  Metric: execution accuracy\n",
        "| Configuration | Overall | Easy | Medium | Hard | Fixed by retry | Avg prompt tokens |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {LABELS[s['config']]} | **{s['accuracy']:.1f}%** | {s['acc_easy']:.0f}% | {s['acc_medium']:.0f}% | "
            f"{s['acc_hard']:.0f}% | {s['fixed_by_retry']} | {s['avg_prompt_tokens']:.0f} |")
    table = "\n".join(lines)
    (out_dir / "summary.md").write_text(table + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    print("\n" + table)
    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
