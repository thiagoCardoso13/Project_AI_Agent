"""Intelligence: chat with your data (text → SQL). Streamlit interface."""
from __future__ import annotations

import html
import re
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.db import list_tables, load_database
from agent.graph import build_graph
from agent.llm import DEFAULTS, get_fallback_llms, get_llm, get_setting
from agent.tools import SessionContext, build_figure

st.set_page_config(page_title="Intelligence · Chat with your data", page_icon="📊", layout="wide")

MAX_QUESTIONS = int(get_setting("MAX_QUESTIONS", "10") or 10)
CSS = (Path(__file__).parent / "assets" / "style.css").read_text()

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

RUNNING_LABELS = {
    "list_tables": "Looking at the available tables…",
    "describe_table": "Reading a table's structure…",
    "execute_sql": "Running SQL…",
    "create_chart": "Building the chart…",
}

_ERROR_RE = re.compile(r"^(SQL |SECURITY )?ERROR", re.IGNORECASE)
_FOLLOWUPS_RE = re.compile(r"(?im)^\s*FOLLOW_?UPS?:\s*(.+)$")


# ---------------------------------------------------------------- data / session
@st.cache_resource(show_spinner="Loading the CSVs…")
def get_connection():
    return load_database()


def init_session():
    if "ctx" in st.session_state:
        return
    st.session_state.ctx = SessionContext(get_connection())
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []
    st.session_state.questions = 0
    st.session_state.graph = None


def get_graph():
    if st.session_state.graph is None:
        st.session_state.graph = build_graph(
            get_llm(), st.session_state.ctx, fallbacks=get_fallback_llms()
        )
    return st.session_state.graph


def new_conversation():
    for k in ("ctx", "thread_id", "history", "questions", "graph"):
        st.session_state.pop(k, None)


def set_pending(question: str):
    st.session_state.pending = question


# ---------------------------------------------------------------- answer parsing
def split_answer(text: str) -> tuple[str, str, list[str]]:
    """Split the model's answer into (headline, body, follow-up questions)."""
    followups: list[str] = []
    m = _FOLLOWUPS_RE.search(text)
    if m:
        followups = [q.strip(" -•*") for q in m.group(1).split("|") if q.strip(" -•*")][:3]
        text = (text[: m.start()] + text[m.end():]).strip()
    parts = text.strip().split("\n\n", 1)
    head = parts[0].strip()
    if len(head) <= 220 and "|" not in head and "\n" not in head and not head.startswith(("#", "-", "*")):
        return head.strip("*# "), (parts[1].strip() if len(parts) > 1 else ""), followups
    return "", text.strip(), followups


def content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)


def tool_note(name: str, args: dict, out: str) -> str:
    if _ERROR_RE.match(out):
        return out.split("\n")[0].split(":", 1)[-1].strip()[:110]
    if name == "list_tables":
        return f"{max(len(out.strip().splitlines()) - 1, 0)} tables found"
    if name == "describe_table":
        return str(args.get("table_name", ""))
    if name == "execute_sql":
        return out.split("\n", 1)[0].replace(".", "")
    if name == "create_chart":
        return f"{args.get('chart_type', '')} chart"
    return ""


# ---------------------------------------------------------------- rendering
def esc(x) -> str:
    return html.escape(str(x), quote=True)


def timeline_html(steps: list[dict], total_s: float) -> str:
    if not steps:
        return ""
    errors = sum(1 for s in steps if s["kind"] == "er")
    n_tools = sum(1 for s in steps if s["kind"] != "fx")
    sub = f"{total_s:.1f}s" if not errors else f"fixed {errors} error{'s' if errors > 1 else ''} · {total_s:.1f}s"
    icon = {"ok": "✓", "er": "✕", "fx": "↻"}
    rows = "".join(
        f'<li class="{s["kind"]}"><span class="dot">{icon[s["kind"]]}</span>'
        f'<span>{("<code>" + esc(s["name"]) + "</code> ") if s["name"] else ""}<span class="note">{esc(s["note"])}</span></span>'
        f'<span class="ms">{esc(str(s["ms"]) + "ms") if s["ms"] is not None else ""}</span></li>'
        for s in steps
    )
    summary = f'<span class="okc">✓</span> Used {n_tools} tool{"s" if n_tools != 1 else ""} <span class="sub">· {sub}</span>'
    return f'<details class="tools"><summary>{summary}</summary><ol class="tl">{rows}</ol></details>'


def render_artifact(msg: dict, idx: int):
    """Chart / Table / SQL tabs inside a card."""
    figs = msg.get("figs") or []
    df = msg.get("df")
    sql = msg.get("sql")
    if not (figs or df is not None or sql):
        return
    title = figs[0][1] if figs else "Result"
    with st.container(key=f"art_{idx}"):
        rows = f"<small>· {len(df)} rows</small>" if df is not None else ""
        st.markdown(f'<div class="arthead">📊 <b>{esc(title)}</b>{rows}</div>', unsafe_allow_html=True)
        names = (["Chart"] if figs else []) + (["Table"] if df is not None else []) + (["SQL"] if sql else [])
        for tab, name in zip(st.tabs(names), names):
            with tab:
                if name == "Chart":
                    for j, (fig, _) in enumerate(figs):
                        st.plotly_chart(fig, width="stretch", key=f"fig_{idx}_{j}", config={"displayModeBar": False})
                elif name == "Table":
                    st.dataframe(df, width="stretch", hide_index=True)
                    if msg.get("truncated"):
                        st.caption("Result limited to the first rows.")
                    st.download_button("Download CSV", df.to_csv(index=False), "result.csv", "text/csv", key=f"dl_{idx}")
                else:
                    st.code(sql, language="sql")


def render_message(msg: dict, idx: int, is_last: bool):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(
                f'<div class="who"><b>You</b><small>{esc(msg["time"])}</small></div><p class="uq">{esc(msg["content"])}</p>',
                unsafe_allow_html=True,
            )
        return
    with st.chat_message("assistant"):
        st.markdown(
            f'<div class="who"><b>Intelligence Agent</b><span class="role">AI</span><small>{esc(msg["time"])}</small></div>'
            + timeline_html(msg.get("steps", []), msg.get("total_s", 0))
            + (f'<p class="lead">{esc(msg["headline"])}</p>' if msg.get("headline") else ""),
            unsafe_allow_html=True,
        )
        render_artifact(msg, idx)
        if msg.get("body"):
            st.markdown(msg["body"])
        if is_last and msg.get("followups"):
            cols = st.columns(len(msg["followups"]))
            for i, (col, q) in enumerate(zip(cols, msg["followups"])):
                col.button(q, key=f"fu_{idx}_{i}", on_click=set_pending, args=(q,), use_container_width=True)


# ---------------------------------------------------------------- the agent run
def ask(question: str):
    ctx: SessionContext = st.session_state.ctx
    now = datetime.now().strftime("%H:%M")
    user_msg = {"role": "user", "content": question, "time": now}
    st.session_state.history.append(user_msg)
    st.session_state.questions += 1
    render_message(user_msg, len(st.session_state.history) - 1, False)

    n_queries, n_charts = len(ctx.queries), len(ctx.charts)
    steps: list[dict] = []
    calls: dict[str, dict] = {}
    answer = ""
    started = time.perf_counter()

    with st.chat_message("assistant"):
        st.markdown(
            f'<div class="who"><b>Intelligence Agent</b><span class="role">AI</span><small>{now}</small></div>',
            unsafe_allow_html=True,
        )
        status = st.status("Thinking…", expanded=False)
        try:
            graph = get_graph()
            config = {"configurable": {"thread_id": st.session_state.thread_id}, "recursion_limit": 30}
            for update in graph.stream({"messages": [HumanMessage(question)]}, config, stream_mode="updates"):
                for _node, data in update.items():
                    for m in (data or {}).get("messages", []):
                        if isinstance(m, AIMessage):
                            for call in m.tool_calls:
                                calls[call["id"]] = call
                                status.update(label=RUNNING_LABELS.get(call["name"], "Working…"))
                            if not m.tool_calls and m.content:
                                answer = content_to_text(m.content)
                        elif isinstance(m, ToolMessage):
                            call = calls.get(m.tool_call_id, {"name": m.name or "tool", "args": {}})
                            out = str(m.content)
                            failed = bool(_ERROR_RE.match(out))
                            if not failed and steps and steps[-1]["kind"] == "er" and call["name"] == "execute_sql":
                                steps.append({"kind": "fx", "name": "", "note": "Rewrote the query and retried", "ms": None})
                            steps.append(
                                {
                                    "kind": "er" if failed else "ok",
                                    "name": call["name"],
                                    "note": tool_note(call["name"], call.get("args", {}), out),
                                    "ms": m.additional_kwargs.get("ms"),
                                }
                            )
            status.update(label="Done", state="complete")
        except Exception as e:  # API quota, invalid key, network…
            status.update(label="Error", state="error")
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                answer = (
                    "⏳ The free AI API quota is exhausted right now (per-minute or daily limit). "
                    "Wait about a minute and try again."
                )
            else:
                answer = f"⚠️ I couldn't answer right now: `{type(e).__name__}`. {str(e)[:300]}"

    headline, body, followups = split_answer(answer or "(no answer)")
    new_queries = ctx.queries[n_queries:]
    last = new_queries[-1] if new_queries else None
    st.session_state.history.append(
        {
            "role": "assistant",
            "time": datetime.now().strftime("%H:%M"),
            "headline": headline,
            "body": body,
            "followups": followups,
            "steps": steps,
            "total_s": time.perf_counter() - started,
            "sql": "\n\n".join(q.sql for q in new_queries) or None,
            "df": last.df if last else None,
            "truncated": last.truncated if last else False,
            "figs": [(build_figure(s, d), s.title) for s, d in ctx.charts[n_charts:]],
        }
    )
    st.rerun()


# ---------------------------------------------------------------- layout
def sidebar():
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="spark">I</div><b>Intelligence</b><span class="tag">Analytics</span></div>',
            unsafe_allow_html=True,
        )
        st.button("＋  New analysis", type="primary", use_container_width=True, on_click=new_conversation)

        tables = list_tables(get_connection())
        rows = "".join(
            f'<div class="src"><i></i><span class="nm">{esc(r.table)}</span><span class="count">{r.rows:,}</span></div>'
            for r in tables.itertuples()
        )
        st.markdown(f'<div class="group">Data sources</div>{rows}', unsafe_allow_html=True)
        st.caption("To use other data, drop a CSV into `demo_data/`.")

        st.markdown('<div class="group">Try asking</div>', unsafe_allow_html=True)
        for i, q in enumerate(EXAMPLES):
            st.button(q, key=f"ex_{i}", on_click=set_pending, args=(q,), use_container_width=True)

        asked = [m["content"] for m in st.session_state.history if m["role"] == "user"]
        if asked:
            items = "".join(
                f'<span class="{"on" if i == len(asked) - 1 else ""}">{esc(q)}</span>' for i, q in enumerate(asked)
            )
            st.markdown(f'<div class="group">This session</div><div class="hist">{items}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="quota">Questions: {st.session_state.questions}/{MAX_QUESTIONS}</div>', unsafe_allow_html=True)


def topbar():
    provider = (get_setting("LLM_PROVIDER", "gemini") or "gemini").lower()
    model = get_setting("LLM_MODEL", DEFAULTS.get(provider)) or provider
    asked = [m["content"] for m in st.session_state.history if m["role"] == "user"]
    title = (asked[0][:60] + ("…" if len(asked[0]) > 60 else "")) if asked else "New analysis"
    n_tables = len(list_tables(get_connection()))
    st.markdown(
        f'<div class="topbar"><div class="crumb">Analyses<span class="sep">/</span><b>{esc(title)}</b></div>'
        f'<div class="model">{esc(model)}</div></div>'
        f'<div class="ctxbar"><span><i class="live"></i>Connected to <b style="font-weight:500;color:#57534c">&nbsp;demo_data/</b></span>'
        f"<span>DuckDB · read-only</span><span>{n_tables} tables loaded</span></div>",
        unsafe_allow_html=True,
    )


def hero():
    st.markdown(
        '<div class="hero"><h1>Ask your data <em>anything.</em></h1>'
        "<p>Ask in plain language. The agent picks the tables, writes the SQL, runs it, "
        "shows the chart and explains the result. Fictional demo data.</p></div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLES[:6]):
        cols[i % 2].button(q, key=f"hero_{i}", on_click=set_pending, args=(q,), use_container_width=True)


def main():
    init_session()
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    sidebar()
    topbar()

    question = st.chat_input("Ask a question about your data…") or st.session_state.pop("pending", None)
    history = st.session_state.history
    if not history and not question:
        hero()
    for i, msg in enumerate(history):
        render_message(msg, i, is_last=(i == len(history) - 1))

    if question:
        if st.session_state.questions >= MAX_QUESTIONS:
            st.warning(
                f"Limit of {MAX_QUESTIONS} questions per session in the public demo (to respect the free API quota). "
                "Click “New analysis” to continue."
            )
        else:
            ask(question)


main()
