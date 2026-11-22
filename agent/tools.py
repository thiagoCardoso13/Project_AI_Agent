"""Agent tools: list_tables, describe_table, execute_sql, create_chart."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import duckdb
import pandas as pd
from langchain_core.tools import tool

from .db import describe_table as _describe_table
from .db import list_tables as _list_tables
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
    """State kept outside the messages: results and charts the UI needs to display."""

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
        text += f"... ({len(df) - limit} rows omitted from this text; total {len(df)})"
    return text


def build_tools(ctx: SessionContext) -> list:
    @tool
    def list_tables() -> str:
        """List the loaded tables and how many rows each has. ALWAYS use this first."""
        return _df_to_text(_list_tables(ctx.con))

    @tool
    def describe_table(table_name: str) -> str:
        """Show the columns, types and 3 sample rows of a table. Use it before writing SQL."""
        try:
            info = _describe_table(ctx.con, table_name)
        except ValueError as e:
            return f"ERROR: {e}"
        return (
            f"Columns of {table_name}:\n{_df_to_text(info['columns'], 100)}\n"
            f"Sample:\n{_df_to_text(info['sample'])}"
        )

    @tool
    def execute_sql(query: str) -> str:
        """Run a SELECT query (DuckDB dialect) and return the result.
        Read-only. If it fails, read the message, fix the SQL and try again."""
        try:
            sql = validate_select(query)
            df = ctx.con.execute(sql).fetchdf()
        except UnsafeSQLError as e:
            return f"SECURITY ERROR: {e}"
        except Exception as e:  # DuckDB errors (missing column, syntax, type...)
            return f"SQL ERROR: {e}\nFix the query and try again."

        truncated = len(df) > MAX_ROWS
        if truncated:
            df = df.head(MAX_ROWS)
        ctx.queries.append(QueryRecord(sql=sql, df=df, truncated=truncated))
        note = f"\n(result limited to {MAX_ROWS} rows)" if truncated else ""
        return f"{len(df)} rows.\n{_df_to_text(df)}{note}"

    @tool
    def create_chart(chart_type: str, x_column: str, y_column: str, title: str, color_column: str = "") -> str:
        """Create a chart from the LAST execute_sql result.
        chart_type: 'line' (time series), 'bar' (categories) or 'pie' (parts of a whole, few categories).
        x_column and y_column must exist in the result; color_column (optional) splits series."""
        last = ctx.last_query
        if last is None:
            return "ERROR: run execute_sql before creating a chart."
        if chart_type not in CHART_TYPES:
            return f"ERROR: chart_type must be one of {CHART_TYPES}."
        cols = set(last.df.columns)
        for c in (x_column, y_column) + ((color_column,) if color_column else ()):
            if c not in cols:
                return f"ERROR: column '{c}' is not in the result. Columns: {sorted(cols)}"
        ctx.charts.append(
            (ChartSpec(chart_type, x_column, y_column, title, color_column or None), last.df.copy())
        )
        return "Chart created; it will be shown to the user."

    return [list_tables, describe_table, execute_sql, create_chart]


PALETTE = ["#e2643c", "#2c4a63", "#efa181", "#1f9d6b", "#b9a6ff", "#ddd8cf"]
# With several series the older one is muted and the newest gets the accent color.
SERIES_PALETTE = ["#ddd8cf", "#e2643c", "#efa181", "#2c4a63", "#1f9d6b", "#b9a6ff"]


def build_figure(spec: ChartSpec, df: pd.DataFrame) -> Any:
    import plotly.express as px

    if spec.kind == "line":
        fig = px.line(df, x=spec.x, y=spec.y, color=spec.color, markers=True, color_discrete_sequence=SERIES_PALETTE if spec.color else PALETTE)
        fig.update_traces(line_width=2.5)
    elif spec.kind == "pie":
        fig = px.pie(df, names=spec.x, values=spec.y, color_discrete_sequence=PALETTE)
    else:
        fig = px.bar(df, x=spec.x, y=spec.y, color=spec.color, barmode="group", color_discrete_sequence=SERIES_PALETTE if spec.color else PALETTE)
        try:
            fig.update_traces(marker_cornerradius=4)
        except Exception:
            pass
    fig.update_layout(
        title=None,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", size=12, color="#57534c"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
        height=330,
    )
    fig.update_xaxes(showgrid=False, linecolor="#e6e2da", title_text=None)
    fig.update_yaxes(gridcolor="#efece6", zeroline=False, title_text=None)
    return fig
