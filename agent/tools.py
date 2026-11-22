"""Ferramentas do agente (listar_tabelas, descrever_tabela, executar_sql, criar_grafico)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import duckdb
import pandas as pd
from langchain_core.tools import tool

from .db import describe_table, list_tables
from .safety import MAX_ROWS, UnsafeSQLError, validate_select

CHART_TYPES = ("line", "bar", "pie")


@dataclass
class QueryRecord:
    sql: str
    df: pd.DataFrame
    truncated: bool = False


@dataclass
class ChartSpec:
    kind: str
    x: str
    y: str
    title: str
    color: Optional[str] = None


@dataclass
class SessionContext:
    """Estado fora das mensagens: resultados e gráficos que a interface precisa exibir."""

    con: duckdb.DuckDBPyConnection
    queries: list[QueryRecord] = field(default_factory=list)
    charts: list[tuple[ChartSpec, pd.DataFrame]] = field(default_factory=list)

    @property
    def last_query(self) -> Optional[QueryRecord]:
        return self.queries[-1] if self.queries else None


def _df_to_text(df: pd.DataFrame, limit: int = 30) -> str:
    shown = df.head(limit)
    text = shown.to_csv(index=False)
    if len(df) > limit:
        text += f"... ({len(df) - limit} linhas omitidas neste texto; total {len(df)})"
    return text


def build_tools(ctx: SessionContext) -> list:
    @tool
    def listar_tabelas() -> str:
        """Lista as tabelas carregadas e quantas linhas cada uma tem. Use SEMPRE primeiro."""
        return _df_to_text(list_tables(ctx.con))

    @tool
    def descrever_tabela(nome_tabela: str) -> str:
        """Mostra colunas, tipos e 3 linhas de exemplo de uma tabela. Use antes de escrever SQL."""
        try:
            info = describe_table(ctx.con, nome_tabela)
        except ValueError as e:
            return f"ERRO: {e}"
        return (
            f"Colunas de {nome_tabela}:\n{_df_to_text(info['columns'], 100)}\n"
            f"Exemplo:\n{_df_to_text(info['sample'])}"
        )

    @tool
    def executar_sql(consulta: str) -> str:
        """Executa uma consulta SELECT (dialeto DuckDB) e devolve o resultado.
        Somente leitura. Se der erro, leia a mensagem, corrija o SQL e tente de novo."""
        try:
            sql = validate_select(consulta)
            df = ctx.con.execute(sql).fetchdf()
        except UnsafeSQLError as e:
            return f"ERRO DE SEGURANÇA: {e}"
        except Exception as e:  # erros do DuckDB (coluna inexistente, sintaxe, tipo...)
            return f"ERRO NO SQL: {e}\nCorrija a consulta e tente novamente."

        truncated = len(df) > MAX_ROWS
        if truncated:
            df = df.head(MAX_ROWS)
        ctx.queries.append(QueryRecord(sql=sql, df=df, truncated=truncated))
        note = f"\n(resultado limitado a {MAX_ROWS} linhas)" if truncated else ""
        return f"{len(df)} linhas.\n{_df_to_text(df)}{note}"

    @tool
    def criar_grafico(tipo: str, coluna_x: str, coluna_y: str, titulo: str, coluna_cor: str = "") -> str:
        """Cria um gráfico a partir do ÚLTIMO resultado de executar_sql.
        tipo: 'line' (séries no tempo), 'bar' (categorias) ou 'pie' (partes de um todo, poucas categorias).
        coluna_x e coluna_y devem existir no resultado; coluna_cor (opcional) separa séries."""
        last = ctx.last_query
        if last is None:
            return "ERRO: rode executar_sql antes de criar um gráfico."
        if tipo not in CHART_TYPES:
            return f"ERRO: tipo deve ser um de {CHART_TYPES}."
        cols = set(last.df.columns)
        for c in (coluna_x, coluna_y) + ((coluna_cor,) if coluna_cor else ()):
            if c not in cols:
                return f"ERRO: coluna '{c}' não está no resultado. Colunas: {sorted(cols)}"
        ctx.charts.append(
            (ChartSpec(tipo, coluna_x, coluna_y, titulo, coluna_cor or None), last.df.copy())
        )
        return "Gráfico criado e será exibido ao usuário."

    return [listar_tabelas, descrever_tabela, executar_sql, criar_grafico]


def build_figure(spec: ChartSpec, df: pd.DataFrame) -> Any:
    import plotly.express as px

    if spec.kind == "line":
        fig = px.line(df, x=spec.x, y=spec.y, color=spec.color, markers=True, title=spec.title)
    elif spec.kind == "pie":
        fig = px.pie(df, names=spec.x, values=spec.y, title=spec.title)
    else:
        fig = px.bar(df, x=spec.x, y=spec.y, color=spec.color, title=spec.title, barmode="group")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig
