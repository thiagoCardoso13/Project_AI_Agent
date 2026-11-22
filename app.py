"""Chat com seus dados (texto → SQL): interface Streamlit."""
from __future__ import annotations

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from agent.db import list_tables, load_database
from agent.graph import build_graph
from agent.llm import get_fallback_llms, get_llm, get_setting
from agent.tools import SessionContext, build_figure

st.set_page_config(page_title="Chat com seus dados", page_icon="💬", layout="wide")

MAX_QUESTIONS = int(get_setting("MAX_QUESTIONS", "10") or 10)

EXAMPLES = [
    "Quanto a loja faturou em 2025?",
    "Mostre o faturamento por mês em 2024 e 2025",
    "Qual categoria tem a maior margem de lucro?",
    "Quais são os 10 produtos mais vendidos em quantidade?",
    "Qual a taxa de devolução por categoria?",
    "Qual departamento tem a maior rotatividade e o principal motivo de saída?",
    "Qual canal de anúncio tem o melhor retorno (receita ÷ gasto)?",
    "Quanto gastei com delivery em 2025?",
]

STEP_LABELS = {
    "listar_tabelas": "🔎 olhando as tabelas disponíveis…",
    "descrever_tabela": "📋 lendo a estrutura de uma tabela…",
    "executar_sql": "⚙️ executando SQL…",
    "criar_grafico": "📊 montando o gráfico…",
}


@st.cache_resource(show_spinner="Carregando os CSVs…")
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
            with st.expander("Ver SQL gerado"):
                st.code(msg["sql"], language="sql")
        if msg.get("df") is not None:
            st.dataframe(msg["df"], width="stretch", hide_index=True)
            if msg.get("truncated"):
                st.caption("Resultado limitado às primeiras linhas.")
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
        status = st.status("Pensando…", expanded=True)
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
                        elif node == "validar":
                            status.write("🛡️ SQL bloqueado: só consultas de leitura são permitidas")
            status.update(label="Pronto", state="complete", expanded=False)
        except Exception as e:  # cota da API, chave inválida, rede…
            status.update(label="Erro", state="error")
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                answer = (
                    "⏳ A cota gratuita da API de IA está esgotada no momento (limite por minuto ou por dia). "
                    "Espere cerca de um minuto e tente de novo."
                )
            else:
                answer = f"⚠️ Não consegui responder agora: `{type(e).__name__}`. {str(e)[:300]}"

    new_queries = ctx.queries[n_queries:]
    last = new_queries[-1] if new_queries else None
    msg = {
        "role": "assistant",
        "content": answer or "(sem resposta)",
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
        st.header("📚 Tabelas carregadas")
        st.dataframe(list_tables(get_connection()), hide_index=True, width="stretch")
        st.caption("Para usar outros dados, coloque um CSV na pasta `demo_data/`.")
        st.header("💡 Perguntas de exemplo")
        for q in EXAMPLES:
            if st.button(q, width="stretch", key=f"ex_{q}"):
                clicked = q
        st.divider()
        st.caption(f"Perguntas nesta sessão: {st.session_state.questions}/{MAX_QUESTIONS}")
        if st.button("🗑️ Nova conversa", width="stretch"):
            for k in ("ctx", "thread_id", "history", "questions", "graph"):
                st.session_state.pop(k, None)
            st.rerun()
    return clicked


def main():
    init_session()
    st.title("💬 Chat com seus dados")
    st.caption(
        "Pergunte em linguagem normal. O agente escolhe as tabelas, gera o SQL, executa, "
        "mostra o gráfico e explica o resultado. (Dados fictícios de demonstração.)"
    )

    example = sidebar()

    for msg in st.session_state.history:
        render_message(msg)

    question = st.chat_input("Ex.: quais produtos mais venderam em novembro?") or example
    if question:
        if st.session_state.questions >= MAX_QUESTIONS:
            st.warning(
                f"Limite de {MAX_QUESTIONS} perguntas por sessão na demo pública (para respeitar a cota gratuita da API). "
                "Clique em “Nova conversa” para continuar."
            )
        else:
            ask(question)


main()
