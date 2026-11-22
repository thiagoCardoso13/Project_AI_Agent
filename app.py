"""Chat With Your Data (text → SQL): Streamlit interface."""
from __future__ import annotations

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from agent.db import list_tables, load_database
from agent.graph import build_graph
from agent.llm import get_fallback_llms, get_llm, get_setting
from agent.tools import SessionContext, build_figure

st.set_page_config(page_title="Chat With Your Data", page_icon="💬", layout="wide")

MAX_QUESTIONS = int(get_setting("MAX_QUESTIONS", "10") or 10)

EXAMPLES = [
    "How much revenue did the store make in 2025?",
    "Show monthly revenue for 2024 and 2025",
    "Which category has the highest profit margin?",
    "What are the 10 best-selling products by quantity?",
    "What is the return rate by category?",
    "Which department has the highest turnover and what is the main exit reason?",
    "Which ad channel has the best return (revenue ÷ spend)?",
    "How much did I spend on food delivery in 2025?",
]

STEP_LABELS = {
    "list_tables": "🔎 looking at the available tables…",
    "describe_table": "📋 reading a table's structure…",
    "execute_sql": "⚙️ running SQL…",
    "create_chart": "📊 building the chart…",
}


@st.cache_resource(show_spinner="Loading the CSVs…")
def get_connection():
    return load_database()


def init_session():
    if "ctx" in st.session_state:
        return
    st.session_state.ctx = SessionContext(get_connection())
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []  # lista de dicts: role, content, sql, df, figs
    st.session_state.questions = 0
    st.session_state.graph = None


def get_graph():
    if st.session_state.graph is None:
        st.session_state.graph = build_graph(get_llm(), st.session_state.ctx, fallbacks=get_fallback_llms())
    return st.session_state.graph


def render_message(msg: dict):
    with st.chat_message(msg["role"]):
        for step in msg.get("steps", []):
            st.caption(step)
        if msg.get("sql"):
            with st.expander("View generated SQL"):
                st.code(msg["sql"], language="sql")
        if msg.get("df") is not None:
            st.dataframe(msg["df"], width="stretch", hide_index=True)
            if msg.get("truncated"):
                st.caption("Result limited to the first rows.")
        for fig in msg.get("figs", []):
            st.plotly_chart(fig, width="stretch")
        st.markdown(msg["content"])


def ask(question: str):
    ctx: SessionContext = st.session_state.ctx
    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.questions += 1
    render_message(st.session_state.history[-1])

    n_queries, n_charts = len(ctx.queries), len(ctx.charts)
    steps: list[str] = []
    answer = ""

    with st.chat_message("assistant"):
        status = st.status("Thinking…", expanded=True)
        try:
            graph = get_graph()
            config = {"configurable": {"thread_id": st.session_state.thread_id}, "recursion_limit": 30}
            for update in graph.stream(
                {"messages": [HumanMessage(question)]}, config, stream_mode="updates"
            ):
                for node, data in update.items():
                    for m in (data or {}).get("messages", []):
                        if isinstance(m, AIMessage):
                            for call in m.tool_calls:
                                label = STEP_LABELS.get(call["name"], call["name"])
                                steps.append(label)
                                status.write(label)
                            if not m.tool_calls and m.content:
                                answer = m.content if isinstance(m.content, str) else "".join(
                                    p.get("text", "") if isinstance(p, dict) else str(p) for p in m.content
                                )
                        elif node == "validate":
                            status.write("🛡️ SQL blocked: only read-only queries are allowed")
            status.update(label="Done", state="complete", expanded=False)
        except Exception as e:  # cota da API, chave inválida, rede…
            status.update(label="Error", state="error")
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                answer = (
                    "⏳ The free AI API quota is exhausted right now (per-minute or daily limit). "
                    "Wait about a minute and try again."
                )
            else:
                answer = f"⚠️ I couldn't answer right now: `{type(e).__name__}`. {str(e)[:300]}"

    new_queries = ctx.queries[n_queries:]
    last = new_queries[-1] if new_queries else None
    msg = {
        "role": "assistant",
        "content": answer or "(no answer)",
        "steps": steps,
        "sql": "\n\n".join(q.sql for q in new_queries) or None,
        "df": last.df if last else None,
        "truncated": last.truncated if last else False,
        "figs": [build_figure(s, d) for s, d in ctx.charts[n_charts:]],
    }
    st.session_state.history.append(msg)
    st.rerun()


def sidebar() -> str | None:
    clicked = None
    with st.sidebar:
        st.header("📚 Loaded tables")
        st.dataframe(list_tables(get_connection()), hide_index=True, width="stretch")
        st.caption("To use other data, drop a CSV into the `demo_data/` folder.")
        st.header("💡 Example questions")
        for q in EXAMPLES:
            if st.button(q, width="stretch", key=f"ex_{q}"):
                clicked = q
        st.divider()
        st.caption(f"Questions this session: {st.session_state.questions}/{MAX_QUESTIONS}")
        if st.button("🗑️ New conversation", width="stretch"):
            for k in ("ctx", "thread_id", "history", "questions", "graph"):
                st.session_state.pop(k, None)
            st.rerun()
    return clicked


def main():
    init_session()
    st.title("💬 Chat With Your Data")
    st.caption(
        "Ask in plain language. The agent picks the tables, writes the SQL, runs it, "
        "shows the chart and explains the result. (Fictional demo data.)"
    )

    example = sidebar()

    for msg in st.session_state.history:
        render_message(msg)

    question = st.chat_input("E.g.: which products sold the most in November?") or example
    if question:
        if st.session_state.questions >= MAX_QUESTIONS:
            st.warning(
                f"Limit of {MAX_QUESTIONS} questions per session in the public demo (to respect the free API quota). "
                "Click “New conversation” to continue."
            )
        else:
            ask(question)


main()
