# datus/tools/verl_tools/bird_text2sql_reward_fn.py
import json
import logging
import math
import os
from typing import Any, Optional, Dict, List
from verl.tools.db_tool import _connector
import sqlglot
from pandas import DataFrame, Series, isna
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))
CONFIG_PATH = "~/bird_dev/dev_20240627/dev_databases/**/*.sqlite"
def compute_score(
        data_source: str,
        solution_str: str,
        ground_truth: str,
        extra_info: Optional[dict[str, Any]] = None,
) -> float:
    """
    Custom reward function for bird_text2sql.

    Parameters
    ----------
    data_source : str
        The dataset name, should be "bird_text2sql".
    solution_str : str
        The decoded model response (full text, including ### Final SQL: ...).
    ground_truth : str
        Ground truth string stored in parquet, e.g. a JSON string
        with fields like {"gold_sql": "...", "db_id": "..."}.
    extra_info : dict | None
        Additional info passed from DataProto.non_tensor_batch["extra_info"].
        If you在agent_loop中把tool_reward塞进extra_info，这里可以用它。

    Returns
    -------
    Final reward in [0.0, 1.0].

    Reward tiers:
        1. Tables do not match between gold_sql and pred_sql               -> 0.0
        2. Tables match but row counts differ                              -> 0.2
        3. Tables + row counts match, but some required columns are missing
        (there is partial correct column overlap)                      -> 0.5
        4. All required columns are present and values match, but there
        are extra columns in prediction                                -> 0.8
        5. Tables, rows, columns, and values are all exactly matched      -> 1.0
    """
    if "### SQL Query:" in solution_str:
        pred_sql = solution_str.split("### SQL Query:")[-1].strip()
    else:
        print(f"🥇Generated SQL not found，solution_str={solution_str}")
        return 0.0
    if pred_sql.startswith("```sql"):
        pred_sql = pred_sql[len("```sql"):].strip()
    elif pred_sql.startswith("```"):
        pred_sql = pred_sql[len("```"):].strip()
    # If you write tool_reward into extra_info in agent_loop/data preprocessing, you can use it directly here:
    if extra_info is not None and extra_info.get("gold_sql")  and extra_info.get("db_id"):
        gold_sql =extra_info.get("gold_sql")
        db_id = extra_info.get("db_id")
    else:
        print(f"🥇extra_info illegal, extra_info={extra_info}")
        return 0.0

    connector = _connector(db_id, CONFIG_PATH)
    gold_exec = connector.execute_pandas(gold_sql)
    pred_exec = connector.execute_pandas(pred_sql)
    if pred_exec is None:
        print("🥇The generated SQL execution returns empty, SQL：", pred_sql)
        return 0.0
    if gold_exec is None:
        print("🥇Standard SQL execution returns null, SQL:", pred_sql)
        return 0.0
    if not gold_exec.success or not isinstance(gold_exec.sql_return, DataFrame):
        print("🥇标Quasi-SQL execution fails or the returned result is not a DataFrame:", gold_exec.error)
        return 0.0
    if not pred_exec.success or not isinstance(pred_exec.sql_return, DataFrame):
        print("🥇The generated SQL execution fails or the returned result is not a DataFrame:", pred_exec.error)
        return 0.0

    # 1) Check table set equality based on SQL text.
    gold_tables = extract_table_names(gold_sql)
    pred_tables = extract_table_names(pred_sql)
    if gold_tables and pred_tables:
        if gold_tables != pred_tables:
            # table does not match -> 0.0
            return 0.0
    gold_df = gold_exec.sql_return
    pred_df = pred_exec.sql_return

    # 2) Check row count.
    gold_rows = len(gold_df)
    pred_rows = len(pred_df)
    if gold_rows != pred_rows:
        # Tables match, but rows are inconsistent -> 0.2
        return 0.2

    compare_result = compare_pandas_tables(pred_df, gold_df)
    if len(compare_result.get("matched_columns", [])) == 0:
        return 0.2
    if len(compare_result.get("missing_columns", [])) > 0:
        return 0.5
    if len(compare_result.get("extra_columns")) > 0:
        return 0.8
    return 1.0

def compare_pandas_tables(actual_df: DataFrame, gold_df: DataFrame) -> Dict[str, Any]:
    if len(actual_df) != len(gold_df):
        return {
            "match_rate": 0.0,
            "matched_columns": [],
            "missing_columns": [],
            "extra_columns": [],
        }

    matches: list[tuple[str, str]] = []
    matched_pred_cols: set[str] = set()
    unmatched_gold_cols: set[str] = set(gold_df.columns)

    for pred_col in actual_df.columns:
        if pred_col in matched_pred_cols:
            continue
        for gold_col in list(unmatched_gold_cols):
            try:
                if columns_match(actual_df[pred_col], gold_df[gold_col]):
                    matches.append((pred_col, gold_col))
                    matched_pred_cols.add(pred_col)
                    unmatched_gold_cols.discard(gold_col)
                    break
            except Exception:
                continue

    un_matches = list(unmatched_gold_cols)
    extra_columns = [col for col in actual_df.columns if col not in matched_pred_cols]

    total_gold_cols = len(gold_df.columns)
    match_rate = (len(matches) / total_gold_cols) if total_gold_cols > 0 else 1.0

    return {
        "match_rate": round(match_rate, 4),
        "matched_columns": matches,
        "missing_columns": un_matches,
        "extra_columns": extra_columns,
    }

def columns_match(series_a: Series, series_b: Series, tol: float = 1e-6) -> bool:
    if len(series_a) != len(series_b):
        return False

    # Sort both series to ensure order-independent comparison
    series_a_sorted = series_a.sort_values(ignore_index=True)
    series_b_sorted = series_b.sort_values(ignore_index=True)

    for a, b in zip(series_a_sorted, series_b_sorted):
        if isna(a) and isna(b):
            continue
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not math.isclose(float(a), float(b), abs_tol=tol):
                return False
        elif a != b:
            return False
    return True

def extract_table_names(sql, dialect="sqlite", ignore_empty=False) -> List[str]:
    """
    Extract fully qualified table names (database.schema.table) from SQL.
    Returns a list of unique table names with original case preserved.
    Filters out CTE (Common Table Expression) tables.
    """
    # Parse the SQL using sqlglot
    read_dialect = parse_read_dialect(dialect)
    try:
        parsed = sqlglot.parse_one(sql, read=read_dialect, error_level=sqlglot.ErrorLevel.IGNORE)
        if parsed is None:
            return []
    except Exception as e:
        logger.warning(f"Error parsing SQL {sql}, error: {e}")
        return []
    table_names = []

    # Get all CTE names
    cte_names = set()
    for cte in parsed.find_all(sqlglot.expressions.CTE):
        if hasattr(cte, "alias") and cte.alias:
            cte_names.add(cte.alias.lower())

    for tb in parsed.find_all(sqlglot.expressions.Table):
        db = tb.catalog
        schema = tb.db
        table_name = tb.name

        # Skip if the table is a CTE
        if table_name.lower() in cte_names:
            continue
        full_name = []

        if dialect in ["mysql", "oracle", "postgres", "postgresql"]:
            if not ignore_empty or schema:
                full_name.append(schema)
        elif dialect not in ("sqlite",):
            if not ignore_empty or db:
                full_name.append(db)
            if not ignore_empty or schema:
                full_name.append(schema)
        full_name.append(table_name)

        table_names.append(".".join(full_name))

    return list(set(table_names))  # Remove duplicates


def parse_read_dialect(dialect: str = "snowflake") -> str:
    """Map SQL dialect to the appropriate read dialect for sqlglot parsing."""
    db = (dialect or "").strip().lower()
    if db in ("postgres", "postgresql", "redshift", "greenplum"):
        return "postgres"
    if db in ("spark", "databricks", "hive", "starrocks"):
        return "hive"
    if db in ("mssql", "sqlserver"):
        return "tsql"
    return dialect
