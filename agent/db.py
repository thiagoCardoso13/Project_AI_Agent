"""Carrega todos os CSVs de uma pasta no DuckDB (uma tabela por arquivo)."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "demo_data"


def load_database(data_dir: Path | str = DEFAULT_DATA_DIR) -> duckdb.DuckDBPyConnection:
    """Cria um banco em memória com uma tabela (view) para cada CSV da pasta.

    O nome da tabela é o nome do arquivo sem a extensão, então para trocar ou
    acrescentar dados basta colocar outro CSV na pasta.
    """
    data_dir = Path(data_dir)
    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {data_dir}")

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
        raise ValueError(f"Tabela '{table}' não existe. Tabelas disponíveis: {sorted(valid)}")
    cols = con.execute(f'DESCRIBE "{table}"').fetchdf()[["column_name", "column_type"]]
    sample = con.execute(f'SELECT * FROM "{table}" LIMIT {int(sample_rows)}').fetchdf()
    return {"columns": cols, "sample": sample}
