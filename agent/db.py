"""Load every CSV in a folder into DuckDB (one table per file)."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "demo_data"


def load_database(data_dir: Path | str = DEFAULT_DATA_DIR) -> duckdb.DuckDBPyConnection:
    """Create an in-memory database with one table per CSV in the folder.

    The table name is the file name without the extension, so to swap or add
    data you just drop another CSV into the folder.
    """
    data_dir = Path(data_dir)
    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    con = duckdb.connect(":memory:")
    for csv in csvs:
        table = csv.stem.lower().replace("-", "_").replace(" ", "_")
        path = str(csv).replace("'", "''")
        con.execute(
            f'CREATE TABLE "{table}" AS SELECT * FROM read_csv_auto(\'{path}\', header=true)'
        )
    return con


def list_tables(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    names = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    rows = [
        {"table": n, "rows": con.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0]}
        for n in names
    ]
    return pd.DataFrame(rows)


def describe_table(con: duckdb.DuckDBPyConnection, table: str, sample_rows: int = 3) -> dict:
    valid = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if table not in valid:
        raise ValueError(f"Table '{table}' does not exist. Available tables: {sorted(valid)}")
    cols = con.execute(f'DESCRIBE "{table}"').fetchdf()[["column_name", "column_type"]]
    sample = con.execute(f'SELECT * FROM "{table}" LIMIT {int(sample_rows)}').fetchdf()
    return {"columns": cols, "sample": sample}
