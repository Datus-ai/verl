# datus/tools/verl_tools/bird_text2sql_reward_fn.py
import logging
import math
import os
from typing import Any, Optional, Dict, List
from .db_tool import _connector
import sqlglot
from pandas import DataFrame, Series, isna
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))
CONFIG_PATH = "~/bird_dev/dev_20240627/dev_databases/**/*.sqlite"

def is_sql_solution(solution_str: str) -> bool:
    """Check if the string starts with SQL keywords."""
    sql_keywords = ["SELECT", "WITH"]
    tokens = solution_str.strip().split()
    if not tokens:
        return False
    first_token = tokens[0].upper()
    return first_token in sql_keywords

def _clean_sql_artifacts(sql: str) -> str:
    """
    Clean up common artifacts from extracted SQL.

    Removes:
    - Tool tags (<tool_call>, <tool_result>, etc.)
    - Markdown code blocks (```sql), including nested ones
    - Trailing whitespace
    """
    # Remove any remaining tool tags
    sql = sql.replace("<tool_call>", "").replace("</tool_call>", "")
    sql = sql.replace("<tool_result>", "").replace("</tool_result>", "").strip()

    # Remove markdown code blocks (loop to handle nested blocks)
    while True:
        original = sql
        # Remove from start
        if sql.startswith("```sql"):
            sql = sql[len("```sql"):].strip()
        elif sql.startswith("```"):
            sql = sql[len("```"):].strip()
        # Remove from end
        if sql.endswith("```"):
            sql = sql[:-3].strip()
        # Break if no change (no more markdown to remove)
        if sql == original:
            break

    return sql

def _truncate_at_terminators(sql: str) -> str:
    """
    Truncate SQL at common termination markers to avoid trailing garbage.

    Terminators: </think>, <think>, tool {, <tool_call>, multiple newlines
    """
    terminators = ["</think>", "<think>", "tool {", "<tool_call>", "\n\n\n"]
    for terminator in terminators:
        if terminator in sql:
            sql = sql.split(terminator)[0].strip()
    return sql

def _find_sql_in_content(content: str) -> Optional[str]:
    """
    Find SQL statement in content that may contain JSON or other non-SQL lines.
    Returns the first line that starts with SQL keywords, or the entire content if it starts with SQL.
    """
    # First try the entire content
    if is_sql_solution(content):
        return content

    # Try line by line
    for line in content.split('\n'):
        stripped = line.strip()
        if stripped and is_sql_solution(stripped):
            # Found a SQL line, collect from here to the end
            remaining_lines = content.split(stripped, 1)[1] if stripped in content else ""
            return stripped + remaining_lines

    return None

def _extract_after_marker(cleaned_str: str, marker: str, label: str) -> Optional[str]:
    """
    Extract SQL content after a marker (e.g., </tool_call>, </tool_result>).

    Args:
        cleaned_str: Pre-cleaned input string
        marker: The marker to split on
        label: Label for logging

    Returns:
        Extracted SQL or None if not found
    """
    content_after = cleaned_str.split(marker)[-1].strip()
    pred_sql = _find_sql_in_content(content_after)

    if pred_sql:
        logger.info(f"Extracted SQL after {label}")
        return pred_sql
    else:
        logger.warning(f"No valid SQL after {marker}, content={content_after[:100]}")
        return None

def extract_sql_from_solution(solution_str: str) -> Optional[str]:
    """
    Extract SQL query from model's solution string.

    Handles various formats:
    1. ### SQL Query: SELECT ...
    2. <tool_call>...</tool_call> SELECT ...
    3. Direct SQL output

    Returns None if no valid SQL found.

    Example:
        >>> solution = "### SQL Query: SELECT * FROM users"
        >>> extract_sql_from_solution(solution)
        'SELECT * FROM users'
    """
    if not solution_str or not solution_str.strip():
        logger.warning("Empty solution_str")
        return None

    # Pre-clean: Remove EOS tokens that might interfere with pattern matching
    cleaned_str = solution_str.replace("<|im_end|>", "").replace("<|endoftext|>", "")

    pred_sql = None

    # Case 1: Model generates "### SQL Query:" header (most reliable)
    if "### SQL Query:" in cleaned_str:
        # Get the last occurrence of ### SQL Query:
        pred_sql = cleaned_str.split("### SQL Query:")[-1].strip()
        # Truncate at termination markers
        pred_sql = _truncate_at_terminators(pred_sql)
        logger.info(f"Extracted SQL after '### SQL Query:' marker")

    # Case 2 & 3: Extract after tool interaction markers
    elif "</tool_result>" in cleaned_str:
        pred_sql = _extract_after_marker(cleaned_str, "</tool_result>", "tool results")
        if not pred_sql:
            return None

    elif "</tool_call>" in cleaned_str:
        pred_sql = _extract_after_marker(cleaned_str, "</tool_call>", "tool calls")
        if not pred_sql:
            return None

    # Case 4: Direct SQL output
    elif is_sql_solution(cleaned_str):
        pred_sql = cleaned_str.strip()
        logger.info(f"Using entire cleaned_str as SQL (starts with SQL keyword)")

    else:
        logger.warning(f"Generated SQL not found, solution_str={cleaned_str[:200]}")
        return None

    # Clean up artifacts
    pred_sql = _clean_sql_artifacts(pred_sql)

    # Final validation
    if not pred_sql:
        logger.warning("pred_sql is empty after cleaning")
        return None

    logger.info(f"Final extracted SQL: {pred_sql[:100]}...")
    return pred_sql

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
        with fields like {"gold_sql": "...", "database": "..."}.
    extra_info : dict | None
        Additional info passed from DataProto.non_tensor_batch["extra_info"].
        Can include tool_call_errors for penalizing bad tool usage.

    Returns
    -------
    Final reward in [0.0, 1.0].

    Reward tiers:
        - Tool call errors (invalid params)                                -> penalty -0.1 per error
        1. Tables do not match between gold_sql and pred_sql               -> 0.0
        2. Tables match but row counts differ                              -> 0.2
        3. Tables + row counts match, but some required columns are missing
        (there is partial correct column overlap)                      -> 0.5
        4. All required columns are present and values match, but there
        are extra columns in prediction                                -> 0.8
        5. Tables, rows, columns, and values are all exactly matched      -> 1.0
    """
    # Check for tool call errors and apply penalty
    tool_penalty = 0.0
    if extra_info and "tool_call_errors" in extra_info:
        num_errors = extra_info["tool_call_errors"]
        tool_penalty = min(0.5, num_errors * 0.2)  # Max 0.5 penalty
        logger.info(f"Tool call errors detected: {num_errors}, penalty: {tool_penalty}")

    # Extract SQL from model output using dedicated function
    pred_sql = extract_sql_from_solution(solution_str)
    if pred_sql is None:
        print(f"🥇Generated SQL not found，solution_str: $$$$ \n", solution_str, "\n END$$$$")
        return 0.0
    print("$$$$ PARSED SQL", pred_sql, "\n$$$$")
    # If you write tool_reward into extra_info in agent_loop/data preprocessing, you can use it directly here:
    if extra_info is not None and extra_info.get("gold_sql")  and extra_info.get("database"):
        gold_sql =extra_info.get("gold_sql")
        db_id = extra_info.get("database")
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

    # Determine base reward based on SQL correctness
    if len(compare_result.get("matched_columns", [])) == 0:
        base_reward = 0.2
    elif len(compare_result.get("missing_columns", [])) > 0:
        base_reward = 0.5
    elif len(compare_result.get("extra_columns")) > 0:
        base_reward = 0.8
    else:
        base_reward = 1.0

    # Apply tool penalty (ensure reward >= 0)
    final_reward = max(0.0, base_reward - tool_penalty)
    return final_reward

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
