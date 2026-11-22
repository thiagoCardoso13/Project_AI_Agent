"""SQL validation: the agent may only read data (SELECT / WITH)."""
from __future__ import annotations

import re

MAX_ROWS = 200

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|attach|detach|"
    r"copy|export|import|install|load|pragma|call|set|grant|revoke|merge|vacuum)\b",
    re.IGNORECASE,
)
_FILE_FUNCS = re.compile(
    r"\b(read_csv\w*|read_parquet|read_json\w*|read_text|read_blob|glob|"
    r"parquet_scan|csv_scan|sniff_csv)\s*\(",
    re.IGNORECASE,
)


class UnsafeSQLError(ValueError):
    pass


def _strip_literals_and_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"'(?:[^']|'')*'", "''", sql)
    sql = re.sub(r'"(?:[^"]|"")*"', '""', sql)
    return sql


def validate_select(sql: str) -> str:
    """Return the cleaned SQL or raise UnsafeSQLError."""
    if not sql or not sql.strip():
        raise UnsafeSQLError("Empty SQL.")

    cleaned = _strip_literals_and_comments(sql).strip().rstrip(";").strip()

    if ";" in cleaned:
        raise UnsafeSQLError("Only one statement at a time is allowed.")
    if not re.match(r"^(select|with)\b", cleaned, re.IGNORECASE):
        raise UnsafeSQLError("Only SELECT queries (or WITH ... SELECT) are allowed.")
    if (m := _FORBIDDEN.search(cleaned)):
        raise UnsafeSQLError(f"Forbidden command in SQL: {m.group(1).upper()}. The agent is read-only.")
    if _FILE_FUNCS.search(cleaned):
        raise UnsafeSQLError("Reading files directly is not allowed; use the loaded tables.")

    return sql.strip().rstrip(";").strip()
