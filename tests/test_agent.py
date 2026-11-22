"""Tests that run without an API key (the model is simulated)."""
from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from agent.db import load_database
from agent.graph import build_graph
from agent.safety import UnsafeSQLError, validate_select
from agent.tools import SessionContext


class FakeToolLLM(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def call(name, args, id_):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id_}])


@pytest.fixture(scope="module")
def con():
    return load_database()


@pytest.mark.parametrize(
    "sql",
    ["SELECT 1", "select * from store_orders;", "WITH a AS (SELECT 1) SELECT * FROM a",
     "SELECT replace(city,'a','b') FROM store_customers", "SELECT 'drop table x' AS t"],
)
def test_select_allowed(sql):
    assert validate_select(sql)


@pytest.mark.parametrize(
    "sql",
    ["DROP TABLE store_orders", "DELETE FROM store_orders", "SELECT 1; DROP TABLE x",
     "UPDATE store_orders SET status='x'", "COPY store_orders TO 'x.csv'",
     "SELECT * FROM read_csv_auto('/etc/passwd')", "ATTACH 'x.db'", "", "PRAGMA database_list"],
)
def test_sql_blocked(sql):
    with pytest.raises(UnsafeSQLError):
        validate_select(sql)


def test_loads_all_tables(con):
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert {"store_orders", "store_customers", "store_products", "store_order_items",
            "hr_employees", "marketing_daily", "support_tickets", "personal_finance_2025"} <= tables


def test_happy_path_with_chart(con):
    ctx = SessionContext(con)
    llm = FakeToolLLM(messages=iter([
        call("list_tables", {}, "1"),
        call("execute_sql", {"query": "SELECT category, COUNT(*) n FROM store_products GROUP BY 1"}, "2"),
        call("create_chart", {"chart_type": "bar", "x_column": "category", "y_column": "n", "title": "Produtos"}, "3"),
        AIMessage(content="There are 6 categories."),
    ]))
    g = build_graph(llm, ctx)
    out = g.invoke({"messages": [HumanMessage("how many products per category?")]},
                   {"configurable": {"thread_id": "t1"}})
    assert out["messages"][-1].content == "There are 6 categories."
    assert len(ctx.queries) == 1 and len(ctx.charts) == 1


def test_dangerous_sql_is_blocked_and_agent_recovers(con):
    ctx = SessionContext(con)
    llm = FakeToolLLM(messages=iter([
        call("execute_sql", {"query": "DROP TABLE store_orders"}, "1"),
        call("execute_sql", {"query": "SELECT COUNT(*) AS n FROM store_orders"}, "2"),
        AIMessage(content="ok"),
    ]))
    g = build_graph(llm, ctx)
    g.invoke({"messages": [HumanMessage("delete everything")]}, {"configurable": {"thread_id": "t2"}})
    assert con.execute("SELECT COUNT(*) FROM store_orders").fetchone()[0] > 0
    assert len(ctx.queries) == 1


def test_gives_up_after_too_many_errors(con):
    ctx = SessionContext(con)
    bad = [call("execute_sql", {"query": "SELECT nope FROM store_orders"}, str(i)) for i in range(5)]
    g = build_graph(FakeToolLLM(messages=iter(bad)), ctx)
    out = g.invoke({"messages": [HumanMessage("x")]}, {"configurable": {"thread_id": "t3"}})
    assert "couldn't reach a valid query" in out["messages"][-1].content
