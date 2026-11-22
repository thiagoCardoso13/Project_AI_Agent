"""Testes que rodam sem chave de API (o modelo é simulado)."""
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
def test_select_permitido(sql):
    assert validate_select(sql)


@pytest.mark.parametrize(
    "sql",
    ["DROP TABLE store_orders", "DELETE FROM store_orders", "SELECT 1; DROP TABLE x",
     "UPDATE store_orders SET status='x'", "COPY store_orders TO 'x.csv'",
     "SELECT * FROM read_csv_auto('/etc/passwd')", "ATTACH 'x.db'", "", "PRAGMA database_list"],
)
def test_sql_bloqueado(sql):
    with pytest.raises(UnsafeSQLError):
        validate_select(sql)


def test_carrega_todas_as_tabelas(con):
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert {"store_orders", "store_customers", "store_products", "store_order_items",
            "hr_employees", "marketing_daily", "support_tickets", "personal_finance_2025"} <= tables


def test_fluxo_feliz_com_grafico(con):
    ctx = SessionContext(con)
    llm = FakeToolLLM(messages=iter([
        call("listar_tabelas", {}, "1"),
        call("executar_sql", {"consulta": "SELECT category, COUNT(*) n FROM store_products GROUP BY 1"}, "2"),
        call("criar_grafico", {"tipo": "bar", "coluna_x": "category", "coluna_y": "n", "titulo": "Produtos"}, "3"),
        AIMessage(content="Há 6 categorias."),
    ]))
    g = build_graph(llm, ctx)
    out = g.invoke({"messages": [HumanMessage("quantos produtos por categoria?")]},
                   {"configurable": {"thread_id": "t1"}})
    assert out["messages"][-1].content == "Há 6 categorias."
    assert len(ctx.queries) == 1 and len(ctx.charts) == 1


def test_sql_perigoso_e_barrado_e_agente_se_corrige(con):
    ctx = SessionContext(con)
    llm = FakeToolLLM(messages=iter([
        call("executar_sql", {"consulta": "DROP TABLE store_orders"}, "1"),
        call("executar_sql", {"consulta": "SELECT COUNT(*) AS n FROM store_orders"}, "2"),
        AIMessage(content="ok"),
    ]))
    g = build_graph(llm, ctx)
    g.invoke({"messages": [HumanMessage("apague tudo")]}, {"configurable": {"thread_id": "t2"}})
    assert con.execute("SELECT COUNT(*) FROM store_orders").fetchone()[0] > 0
    assert len(ctx.queries) == 1


def test_desiste_depois_de_muitos_erros(con):
    ctx = SessionContext(con)
    bad = [call("executar_sql", {"consulta": "SELECT nope FROM store_orders"}, str(i)) for i in range(5)]
    g = build_graph(FakeToolLLM(messages=iter(bad)), ctx)
    out = g.invoke({"messages": [HumanMessage("x")]}, {"configurable": {"thread_id": "t3"}})
    assert "Não consegui" in out["messages"][-1].content
