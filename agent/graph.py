"""O agente como um grafo próprio do LangGraph (StateGraph).

    inicio → entrada → agente ─┬─(sem ferramentas)──────────────→ fim
                               └─(com ferramentas)→ validar → executar ─┬→ agente (loop)
                                                                        └→ desistir → fim
                                  (SQL bloqueado volta direto pro agente)
"""
from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from .safety import UnsafeSQLError, validate_select
from .tools import SessionContext, build_tools

MAX_SQL_ERRORS = 3  # a 1ª tentativa + até 2 autocorreções

SYSTEM_PROMPT = """Você é um analista de dados sênior. Responde perguntas sobre os dados carregados \
consultando-os com SQL (dialeto DuckDB). Responda no mesmo idioma da pergunta.

Método (como um analista):
1. Chame `listar_tabelas` e depois `descrever_tabela` nas tabelas relevantes ANTES de escrever SQL. Nunca invente colunas.
2. Escreva o SQL e rode com `executar_sql`. Se der erro, leia a mensagem, corrija e tente de novo.
3. Se o resultado for adequado para visualização (série temporal → line, categorias → bar, partes de um todo → pie), chame `criar_grafico`.
4. Termine com uma explicação curta e clara, em linguagem simples, com os números principais. Não repita a tabela inteira.

Como as tabelas se ligam: store_orders.customer_id → store_customers; store_order_items.order_id → store_orders; \
store_order_items.product_id → store_products. Demais tabelas são independentes. Valores monetários em euros, período 2024–2025.

Convenções (diga na resposta quando usar uma):
- "Faturamento/vendas" da loja = soma de store_orders.order_total_eur dos pedidos com status NOT IN ('cancelled','returned'). \
Status possíveis: delivered, shipped, processing, returned, cancelled. "Devolução" = status 'returned'.
- Margem de um item = line_total_eur - quantity * cost_eur (cost_eur está em store_products). "Maior margem de lucro" = margem PERCENTUAL (lucro ÷ receita); mostre também o lucro em euros.
- Datas: order_datetime é TIMESTAMP; use year()/month()/date_trunc(). Black Friday = novembro (pico de vendas), \
a menos que o usuário defina o dia.
- Use ROUND(..., 2) em valores monetários e ordene os resultados de forma útil.
- Use apenas SELECT. Qualquer tentativa de alterar dados é bloqueada.
- Se a pergunta for ambígua, escolha a interpretação mais razoável, diga qual foi e responda; não devolva a pergunta ao usuário.
- Se os dados não puderem responder à pergunta, diga isso honestamente."""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    sql_errors: int


def build_graph(llm: Any, ctx: SessionContext, checkpointer: Any = None, fallbacks: Any = ()):
    tools = build_tools(ctx)
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)
    if fallbacks:  # se o modelo principal falhar (ex.: cota), tenta os de reserva
        llm_with_tools = llm_with_tools.with_fallbacks([f.bind_tools(tools) for f in fallbacks])

    def entrada(state: AgentState) -> dict:
        # Novo turno do usuário: zera o contador de erros (a memória das mensagens continua).
        return {"sql_errors": 0}

    def agente(state: AgentState) -> dict:
        response = llm_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
        return {"messages": [response]}

    def depois_do_agente(state: AgentState) -> str:
        last = state["messages"][-1]
        return "validar" if getattr(last, "tool_calls", None) else END

    def validar(state: AgentState) -> dict:
        """Barra qualquer SQL que não seja somente leitura, antes de tocar no banco."""
        blocked = []
        for call in state["messages"][-1].tool_calls:
            if call["name"] != "executar_sql":
                continue
            try:
                validate_select(call["args"].get("consulta", ""))
            except UnsafeSQLError as e:
                blocked.append(
                    ToolMessage(
                        content=f"ERRO DE SEGURANÇA: {e}",
                        tool_call_id=call["id"],
                        name=call["name"],
                    )
                )
        return {"messages": blocked, "sql_errors": state["sql_errors"] + len(blocked)}

    def depois_de_validar(state: AgentState) -> str:
        answered = {m.tool_call_id for m in state["messages"] if isinstance(m, ToolMessage)}
        pending = [c for c in _last_ai(state).tool_calls if c["id"] not in answered]
        if state["sql_errors"] >= MAX_SQL_ERRORS and not pending:
            return "desistir"
        return "executar" if pending else "agente"

    def executar(state: AgentState) -> dict:
        ai = _last_ai(state)
        answered = {m.tool_call_id for m in state["messages"] if isinstance(m, ToolMessage)}
        results, errors = [], 0
        for call in ai.tool_calls:
            if call["id"] in answered:
                continue
            tool = tools_by_name.get(call["name"])
            if tool is None:
                out = f"ERRO: ferramenta '{call['name']}' não existe."
            else:
                try:
                    out = str(tool.invoke(call["args"]))
                except Exception as e:
                    out = f"ERRO ao chamar a ferramenta: {e}"
            if out.startswith("ERRO"):
                errors += 1
            results.append(ToolMessage(content=out, tool_call_id=call["id"], name=call["name"]))
        return {"messages": results, "sql_errors": state["sql_errors"] + errors}

    def depois_de_executar(state: AgentState) -> str:
        return "desistir" if state["sql_errors"] >= MAX_SQL_ERRORS else "agente"

    def desistir(state: AgentState) -> dict:
        return {
            "messages": [
                AIMessage(
                    content="Não consegui chegar a uma consulta válida depois de algumas tentativas. "
                    "Tente reformular a pergunta ou ser mais específico sobre as tabelas/colunas."
                )
            ]
        }

    g = StateGraph(AgentState)
    g.add_node("entrada", entrada)
    g.add_node("agente", agente)
    g.add_node("validar", validar)
    g.add_node("executar", executar)
    g.add_node("desistir", desistir)

    g.add_edge(START, "entrada")
    g.add_edge("entrada", "agente")
    g.add_conditional_edges("agente", depois_do_agente, {"validar": "validar", END: END})
    g.add_conditional_edges("validar", depois_de_validar, {"executar": "executar", "agente": "agente", "desistir": "desistir"})
    g.add_conditional_edges("executar", depois_de_executar, {"agente": "agente", "desistir": "desistir"})
    g.add_edge("desistir", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())


def _last_ai(state: AgentState) -> AIMessage:
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage):
            return m
    raise RuntimeError("Nenhuma mensagem do agente no estado.")
