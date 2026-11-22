"""The agent as a custom LangGraph graph (StateGraph).

    start → intake → agent ─┬─(no tool calls)────────────────→ end
                            └─(tool calls)→ validate → execute ─┬→ agent (loop)
                                                                └→ give_up → end
                               (blocked SQL goes straight back to the agent)
"""
from __future__ import annotations

import re
import time
from typing import Annotated, Any

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from .safety import UnsafeSQLError, validate_select
from .tools import SessionContext, build_tools

MAX_SQL_ERRORS = 3  # the 1st attempt + up to 2 self-corrections

_ERROR_RE = re.compile(r"^(SQL |SECURITY )?ERROR", re.IGNORECASE)

SYSTEM_PROMPT = """You are a senior data analyst. You answer questions about the loaded data by querying it \
with SQL (DuckDB dialect). Reply in the same language as the user's question.

Method (like an analyst):
1. Call `list_tables`, then `describe_table` on the relevant tables BEFORE writing SQL. Never invent columns.
2. Write the SQL and run it with `execute_sql`. If it fails, read the error, fix it and try again.
3. If the result suits a visualization (time series → line, categories → bar, parts of a whole → pie), call `create_chart`.
4. Write the final answer in this exact shape:
   - First line: ONE short headline sentence that states the answer itself with the key number (max ~15 words, plain text, no markdown, no bullet).
   - A blank line, then 1-2 short paragraphs of plain-language explanation (you may use **bold** for key terms). Do NOT repeat the result table or list every row: the user already sees the table and chart.
   - A final line, alone: FOLLOW_UPS: <question 1> | <question 2> | <question 3> (three short follow-up questions the data could answer).

How the tables relate: store_orders.customer_id → store_customers; store_order_items.order_id → store_orders; \
store_order_items.product_id → store_products. The other tables are independent. Money is in euros, period 2024–2025.

Conventions (state which one you used in your answer):
- Store "revenue/sales" = sum of store_orders.order_total_eur for orders with status NOT IN ('cancelled','returned'). \
Possible statuses: delivered, shipped, processing, returned, cancelled. A "return" = status 'returned'.
- Item margin = line_total_eur - quantity * cost_eur (cost_eur lives in store_products). "Highest profit margin" = \
PERCENTAGE margin (profit ÷ revenue); also show the profit in euros.
- Dates: order_datetime is a TIMESTAMP; use year()/month()/date_trunc(). Black Friday = November (sales peak), \
unless the user defines the day.
- When comparing periods (e.g. 2024 vs 2025 by month), return one row per month and year (columns: month, year, value) and chart it with color_column='year' so each year is its own series.
- Use ROUND(..., 2) on money values and order results usefully.
- Use SELECT only. Any attempt to change data is blocked.
- If the question is ambiguous, pick the most reasonable interpretation, say which one, and answer; do not bounce the question back.
- If the data cannot answer the question, say so honestly."""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    sql_errors: int


def build_graph(llm: Any, ctx: SessionContext, checkpointer: Any = None, fallbacks: Any = ()):
    tools = build_tools(ctx)
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)
    if fallbacks:  # if the main model fails (e.g. quota), try the backup models
        llm_with_tools = llm_with_tools.with_fallbacks([f.bind_tools(tools) for f in fallbacks])

    def intake(state: AgentState) -> dict:
        # New user turn: reset the error counter (message memory is kept).
        return {"sql_errors": 0}

    def agent(state: AgentState) -> dict:
        response = llm_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
        return {"messages": [response]}

    def after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        return "validate" if getattr(last, "tool_calls", None) else END

    def validate(state: AgentState) -> dict:
        """Block any SQL that is not read-only, before it touches the database."""
        blocked = []
        for call in state["messages"][-1].tool_calls:
            if call["name"] != "execute_sql":
                continue
            try:
                validate_select(call["args"].get("query", ""))
            except UnsafeSQLError as e:
                blocked.append(
                    ToolMessage(
                        content=f"SECURITY ERROR: {e}",
                        tool_call_id=call["id"],
                        name=call["name"],
                    )
                )
        return {"messages": blocked, "sql_errors": state["sql_errors"] + len(blocked)}

    def after_validate(state: AgentState) -> str:
        answered = {m.tool_call_id for m in state["messages"] if isinstance(m, ToolMessage)}
        pending = [c for c in _last_ai(state).tool_calls if c["id"] not in answered]
        if state["sql_errors"] >= MAX_SQL_ERRORS and not pending:
            return "give_up"
        return "execute" if pending else "agent"

    def execute(state: AgentState) -> dict:
        ai = _last_ai(state)
        answered = {m.tool_call_id for m in state["messages"] if isinstance(m, ToolMessage)}
        results, errors = [], 0
        for call in ai.tool_calls:
            if call["id"] in answered:
                continue
            tool = tools_by_name.get(call["name"])
            started = time.perf_counter()
            if tool is None:
                out = f"ERROR: tool '{call['name']}' does not exist."
            else:
                try:
                    out = str(tool.invoke(call["args"]))
                except Exception as e:
                    out = f"ERROR while calling the tool: {e}"
            if _ERROR_RE.match(out):
                errors += 1
            ms = round((time.perf_counter() - started) * 1000)
            results.append(
                ToolMessage(
                    content=out,
                    tool_call_id=call["id"],
                    name=call["name"],
                    additional_kwargs={"ms": ms},  # shown in the UI's tool timeline
                )
            )
        return {"messages": results, "sql_errors": state["sql_errors"] + errors}

    def after_execute(state: AgentState) -> str:
        return "give_up" if state["sql_errors"] >= MAX_SQL_ERRORS else "agent"

    def give_up(state: AgentState) -> dict:
        return {
            "messages": [
                AIMessage(
                    content="I couldn't reach a valid query after a few attempts. "
                    "Try rephrasing the question or being more specific about the tables/columns."
                )
            ]
        }

    g = StateGraph(AgentState)
    g.add_node("intake", intake)
    g.add_node("agent", agent)
    g.add_node("validate", validate)
    g.add_node("execute", execute)
    g.add_node("give_up", give_up)

    g.add_edge(START, "intake")
    g.add_edge("intake", "agent")
    g.add_conditional_edges("agent", after_agent, {"validate": "validate", END: END})
    g.add_conditional_edges(
        "validate", after_validate, {"execute": "execute", "agent": "agent", "give_up": "give_up"}
    )
    g.add_conditional_edges("execute", after_execute, {"agent": "agent", "give_up": "give_up"})
    g.add_edge("give_up", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())


def _last_ai(state: AgentState) -> AIMessage:
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage):
            return m
    raise RuntimeError("No agent message in the state.")
