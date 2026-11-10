"""Streamlit chat UI.  Run:  streamlit run app.py"""
import os
from pathlib import Path

import plotly.express as px
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from text2sql.agent import AgentConfig, Text2SQLAgent
from text2sql.llm import ChatLLM, LLMError

st.set_page_config(page_title="Olist Text-to-SQL Assistant", page_icon="📊", layout="wide")

# Streamlit renamed `use_container_width=True` to `width="stretch"`; support both.
_ver = tuple(int(x) for x in st.__version__.split(".")[:2] if x.isdigit())
FULL_WIDTH = {"width": "stretch"} if _ver >= (1, 50) else {"use_container_width": True}

EXAMPLE_QUESTIONS = [
    "What was the monthly revenue in 2017?",
    "Which 10 product categories have the highest revenue?",
    "Which customer states have the slowest average delivery?",
    "How does the review score change when an order is delivered late?",
    "What percentage of customers placed more than one order?",
    "Which payment types are most popular, and what is their average value?",
]

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Settings")
    db_path = st.text_input("SQLite database", os.getenv("DB_PATH", "data/olist.db"))
    mode_label = st.selectbox("Schema context", ["Retrieved tables (RAG)", "All tables, documented", "Raw CREATE TABLE"])
    schema_mode = {"Retrieved tables (RAG)": "rag", "All tables, documented": "docs", "Raw CREATE TABLE": "raw"}[mode_label]
    use_fewshot = st.toggle("Few-shot examples", value=True)
    self_correct = st.toggle("Self-correction", value=True)
    summarize = st.toggle("Plain-English answer", value=True)
    st.caption(f"Model: `{os.getenv('LLM_MODEL', 'not set')}`")
    if st.button("Clear chat"):
        st.session_state.messages = []

    st.subheader("Try asking")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q):
            st.session_state.pending = q

st.title("📊 Ask the Olist e-commerce database")
st.caption("Questions in plain English → SQL → answer, table and chart. Read-only access.")

if not Path(db_path).exists():
    st.error(f"Database not found at `{db_path}`. Build it first:\n\n"
             "`python scripts/build_db.py --csv-dir data/olist --db data/olist.db`")
    st.stop()


@st.cache_resource(show_spinner="Loading schema...")
def get_agent(path, mode, fewshot, correct):
    llm = ChatLLM()
    cfg = AgentConfig(schema_mode=mode, use_fewshot=fewshot, self_correct=correct, check_empty=correct)
    return Text2SQLAgent(path, llm, cfg)


try:
    agent = get_agent(db_path, schema_mode, use_fewshot, self_correct)
except LLMError as e:
    st.error(f"{e}\n\nCopy `.env.example` to `.env` and fill in your API key and model.")
    st.stop()


def render(result):
    if result.error:
        st.error(f"I couldn't answer that. Last error: {result.error}")
    else:
        if result.answer:
            st.markdown(result.answer)
        df = result.df
        if result.chart:
            c = result.chart
            plot = px.line if c["kind"] == "line" else px.bar
            fig = plot(df, x=c["x"], y=c["y"], markers=True) if c["kind"] == "line" else plot(df, x=c["x"], y=c["y"], barmode="group")
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
            st.plotly_chart(fig, **FULL_WIDTH)
        st.dataframe(df, hide_index=True, **FULL_WIDTH)
        if result.truncated:
            st.caption(f"Showing the first {len(df):,} rows.")
        st.download_button("Download CSV", df.to_csv(index=False), file_name="result.csv",
                           mime="text/csv", key=f"dl_{id(result)}")
    with st.expander("SQL and details"):
        if result.sql:
            st.code(result.sql, language="sql")
        if len(result.attempts) > 1:
            st.markdown(f"**Self-correction:** {len(result.attempts)} attempts")
            for i, a in enumerate(result.attempts, 1):
                st.markdown(f"Attempt {i}: " + (f"❌ `{a.error}`" if a.error else f"✅ {a.rows} rows"))
        st.caption(f"Tables in context: {', '.join(result.tables)}")
        if result.examples_used:
            st.caption("Similar examples used: " + " | ".join(result.examples_used))
        st.caption(f"Tokens: {result.prompt_tokens:,} in / {result.completion_tokens:,} out")


if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["content"])
        else:
            render(m["result"])

question = st.chat_input("e.g. Which states have the most late deliveries?")
if st.session_state.get("pending"):
    question = st.session_state.pop("pending")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Writing and running SQL..."):
            try:
                result = agent.ask(question, summarize=summarize)
            except LLMError as e:
                st.error(str(e))
                st.stop()
        render(result)
    st.session_state.messages.append({"role": "assistant", "result": result})
