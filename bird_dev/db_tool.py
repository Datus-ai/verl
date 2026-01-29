# verl_tools.py - 专门给 Verl 用的工具入口
import glob
import json
import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List, Literal

import ray


from pandas import DataFrame
from pyarrow import Table
from pydantic import Field, BaseModel

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

TABLE_TYPE = Literal["table", "view", "mv", "full"]



def file_stem_from_uri(uri: str) -> str:
    """
    Extract the stem of the file name (remove extension) from the URI of DuckDB/SQLite or the normal path.
    e.g. duckdb:///path/to/demo.duckdb -> demo
         sqlite:////tmp/foo.db -> foo
         /abs/path/bar.duckdb -> bar
         foo.db -> foo
    """
    if not uri:
        return ""
    try:
        path = uri.split(":///")[-1] if ":///" in uri else uri
        base = os.path.basename(path)
        stem, _ = os.path.splitext(base)
        return stem
    except Exception:
        # reveal all the details
        return uri.split("/")[-1].split(".")[0]



class BaseResult(BaseModel):
    """
    Base class for all node result data validation.
    Provides common validation functionality for node results.
    """

    success: bool = Field(..., description="Indicates whether the operation was successful")
    error: Optional[str] = Field(None, description="Error message if operation failed")

    # Action history and execution stats for agentic nodes
    action_history: Optional[List[dict]] = Field(
        default=None, description="Complete history of tool calls and actions during execution"
    )
    execution_stats: Optional[dict] = Field(
        default=None, description="Execution statistics (tokens, tools called, duration, etc.)"
    )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key with an optional default value."""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access to attributes."""
        return getattr(self, key)

    def to_str(self) -> str:
        """Convert the result to a string representation, including all nested objects."""
        return self.model_dump_json()

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary representation."""
        return self.model_dump()

    @classmethod
    def from_str(cls, json_str: str) -> "BaseResult":
        return cls.model_validate_json(json_str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseInput":
        """Create SqlTask instance from dictionary."""
        return cls.model_validate(data)

    class Config:
        extra = "forbid"  # Prevent extra fields not defined in the model


MAX_SQL_RESULT_LENGTH = int(os.getenv("MAX_SQL_RESULT_LENGTH", 2000))

class ExecuteSQLResult(BaseResult):
    """
    Result model for SQL execution node.
    Contains the execution results.
    """

    sql_query: Optional[str] = Field("", description="The SQL query to execute")
    row_count: Optional[int] = Field(None, description="The number of rows returned")
    sql_return: Any = Field(  # TODO: change to Union[str, ArrowTable, List[Reuslt]]
        default=None, description="The result of SQL execution (string or Arrow data)"
    )
    result_format: str = Field(default="", description="Format of the result: 'csv' or 'arrow' or 'pandas' or 'list'")

    class Config:
        arbitrary_types_allowed = True

    def compact_result(self) -> str:
        """
        Returns a compact string representation of the execution result.
        Only includes row count and truncated sql return (max length defined by DATUS_MAX_RESULT_LENGTH).
        Returns:
            str: Formatted string with row count and truncated result
        """
        sql_result = ""
        if hasattr(self.sql_return, "to_csv"):
            sql_result = self.sql_return.to_csv(index=False)
        else:
            sql_result = str(self.sql_return)
        truncated_return = (
            (sql_result[:MAX_SQL_RESULT_LENGTH] + "...")
            if sql_result and len(sql_result) > MAX_SQL_RESULT_LENGTH
            else sql_result
        )

        # errors = f"Error: {self.error}\n" if not self.success else ""
        return f"Error: {self.error}\nRows: {self.row_count}\nResult: {truncated_return}"

class FuncToolResult(BaseModel):
    success: int = Field(
        default=1, description="Whether the execution is successful or not, 1 is success, 0 is failure", init=True
    )
    error: Optional[str] = Field(
        default=None, description="Error message: field is not empty when success=0", init=True
    )
    result: Optional[Any] = Field(default=None, description="Result of the execution", init=True)

class SQLiteConnector:
    """
    Connector for SQLite databases using native sqlite3 SDK.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path.replace("sqlite:///", "")

        self.database_name = file_stem_from_uri(self.db_path)


    def _handle_exception(self, e: Exception, sql: str = "") -> Exception:
        logger.error(f'Failed execute SQL: \n {sql}',exc_info=e)
        return e


    def execute_insert(self, sql: str) -> ExecuteSQLResult:
        """Execute an INSERT SQL statement."""
        try:
            with get_db_connection(self.db_path) as connection:
                cursor = connection.cursor()
                cursor.execute(sql)
                connection.commit()
            return ExecuteSQLResult(
                success=True,
                sql_query=sql,
                sql_return=str(cursor.lastrowid),
                row_count=cursor.rowcount,
            )
        except Exception as e:
            ex = self._handle_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql,
            )

    def execute_update(self, sql: str) -> ExecuteSQLResult:
        """Execute an UPDATE SQL statement."""
        try:
            with get_db_connection(self.db_path) as connection:
                cursor = connection.cursor()
                cursor.execute(sql)
                connection.commit()
                return ExecuteSQLResult(
                    success=True,
                    sql_query=sql,
                    sql_return=str(cursor.rowcount),
                    row_count=cursor.rowcount,
                )
        except Exception as e:
            ex = self._handle_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql,
            )

    def execute_delete(self, sql: str) -> ExecuteSQLResult:
        """Execute a DELETE SQL statement."""
        try:
            with get_db_connection(self.db_path) as connection:
                cursor = connection.cursor()
                cursor.execute(sql)
                connection.commit()
                return ExecuteSQLResult(
                    success=True,
                    sql_query=sql,
                    sql_return=str(cursor.rowcount),
                    row_count=cursor.rowcount,
                )
        except Exception as e:
            ex = self._handle_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql,
            )

    def execute_ddl(self, sql: str) -> ExecuteSQLResult:
        """Execute a DDL SQL statement."""
        try:
            with get_db_connection(self.db_path) as connection:
                cursor = connection.cursor()
                cursor.execute(sql)
                connection.commit()
                return ExecuteSQLResult(
                    success=True,
                    sql_query=sql,
                    sql_return="Success",
                    row_count=0,
                )
        except Exception as e:
            ex = self._handle_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql,
            )

    def execute_query(
            self, sql: str, result_format: Literal["csv", "arrow", "pandas", "list"] = "csv"
    ) -> ExecuteSQLResult:
        """Execute a SELECT query."""
        try:
            with get_db_connection(self.db_path) as connection:
                cursor = connection.cursor()
                cursor.execute(sql)
                connection.commit()
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description] if cursor.description else []

                # Convert to list of dicts
                result_list = [dict(zip(columns, row)) for row in rows]
                row_count = len(rows)

                # Explicitly pass columns to preserve schema for empty results
                df = DataFrame(result_list, columns=columns)

                if result_format == "csv":
                    result = df.to_csv(index=False)
                elif result_format == "arrow":
                    result = Table.from_pandas(df)
                elif result_format == "pandas":
                    result = df
                else:  # list
                    result = result_list

                return ExecuteSQLResult(
                    success=True,
                    sql_query=sql,
                    sql_return=result,
                    row_count=row_count,
                    result_format=result_format,
                )
        except Exception as e:
            ex = self._handle_exception(e, sql)
            return ExecuteSQLResult(
                success=False,
                error=str(ex),
                sql_query=sql,
            )

    def execute_pandas(self, sql: str) -> ExecuteSQLResult:
        """Execute query and return pandas DataFrame."""
        return self.execute_query(sql, result_format="pandas")

    def execute_csv(self, sql: str) -> ExecuteSQLResult:
        """Execute query and return CSV format."""
        return self.execute_query(sql, result_format="csv")


    def get_tables(self, catalog_name: str = "", database_name: str = "", schema_name: str = "") -> List[str]:
        """Get all table names."""
        with get_db_connection(self.db_path) as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            return [row[0] for row in cursor.fetchall()]

    def get_views(self, catalog_name: str = "", database_name: str = "", schema_name: str = "") -> List[str]:
        """Get all view names."""
        with get_db_connection(self.db_path) as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
            return [row[0] for row in cursor.fetchall()]

    def full_name(
            self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_name: str = ""
    ) -> str:
        return f'"{table_name}"'

    def do_switch_context(self, catalog_name: str = "", database_name: str = "", schema_name: str = ""):
        """SQLite does not support switch context"""

    def _get_schema_with_ddl(
            self, database_name: str = "", table_type: str = "table", filter_tables: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """Get schema with DDL for tables or views."""
        with get_db_connection(self.db_path) as connection:
            cursor = connection.cursor()
            cursor.execute(f"SELECT name, sql FROM sqlite_master WHERE type='{table_type}'")

            schema_list = []
            for row in cursor.fetchall():
                table_name = row[0]
                definition = row[1]

                # Skip SQLite system tables
                if table_name.startswith("sqlite_"):
                    continue

                if filter_tables and table_name not in filter_tables:
                    continue

                schema_list.append(
                    {
                        "identifier": f"{database_name}.{table_name}",
                        "catalog_name": "",
                        "database_name": database_name,
                        "schema_name": "",
                        "table_name": table_name,
                        "definition": definition,
                        "table_type": table_type,
                    }
                )

            return schema_list

    def get_tables_with_ddl(
            self, catalog_name: str = "", database_name: str = "", schema_name: str = "", tables: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """Get tables with DDL definitions."""
        return self._get_schema_with_ddl(
            database_name=database_name or self.database_name,
            table_type="table",
            filter_tables=tables,
        )

    def get_views_with_ddl(
            self, catalog_name: str = "", database_name: str = "", schema_name: str = "", tables: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """Get views with DDL definitions."""
        return self._get_schema_with_ddl(
            database_name=database_name or self.database_name,
            table_type="view", filter_tables=tables
        )

    def get_schema(
            self, catalog_name: str = "", database_name: str = "", schema_name: str = "", table_name: str = ""
    ) -> List[Dict[str, str]]:
        """Get schema information for a table."""
        if not table_name:
            return []

        with get_db_connection(self.db_path) as connection:
            cursor = connection.cursor()
            try:
                cursor.execute(f'PRAGMA table_info("{table_name}")')
                columns = cursor.fetchall()
                return [
                    {
                        "cid": col[0],
                        "name": col[1],
                        "type": col[2],
                        "notnull": col[3],
                        "dflt_value": col[4],
                        "pk": col[5],
                    }
                    for col in columns
                ]
            except Exception as e:
                raise self._handle_exception(e, f'PRAGMA table_info("{table_name}")')
DB_PATH_DICT = {}

def _parse_glob_path(path_pattern: str, dialect:str="sqlite") -> List[Dict[str, str]]:
    path_pattern = os.path.expanduser(path_pattern)
    normalized_pattern = path_pattern.replace("\\", "/")

    # Detect whether the directory part contains any wildcard
    if "/" in normalized_pattern:
        dir_pattern, _ = normalized_pattern.rsplit("/", 1)
    else:
        dir_pattern, _ = "", normalized_pattern
    dir_has_wildcard = any(ch in dir_pattern for ch in ("*", "?", "["))

    files = glob.glob(path_pattern, recursive=True)
    result: List[Dict[str, str]] = []

    for file_path in files:
        path = Path(file_path)
        if not path.is_file():
            continue

        database_name = path.stem  # File name (remove extension)
        if dir_has_wildcard:
            logic_name = path.parent.name
        else:
            logic_name = database_name

        uri = f"{dialect}:///{path.as_posix()}"
        result.append(
            {
                "logic_name": logic_name,
                "name": database_name,
                "uri": uri,
            }
        )
    return result

def db_paths(db_name: str, path_glob: str = "~/bird/dev_20240627/dev_databases/**/*.sqlite") -> str:
    global DB_PATH_DICT
    if DB_PATH_DICT:
        return DB_PATH_DICT.get(db_name)
    for item in _parse_glob_path(path_glob):
        DB_PATH_DICT[item["name"]] = item["uri"]
    return DB_PATH_DICT.get(db_name)

@contextmanager
def get_db_connection(db_path):
    """
    A thread-safe SQLite connection context manager.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.isolation_level = None # Optional: Set to auto-commit mode
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"⚠️ Close connection for SQLite failed：{e}")
    finally:
        if conn:
            conn.close()
def _connector(db_name:str, config_path: str) -> SQLiteConnector:
    db_uri = db_paths(db_name, config_path)
    if not db_uri:
        raise RuntimeError(f"Database {db_name} is not found in configuration")
    return SQLiteConnector(db_path=db_uri)

class BaseVerlDBTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize DBFuncTool with configuration and schema.
        The configuration of namespace multi-database is not currently supported.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition

        Example tool_schema:
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Searches for relevant information based on queries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_list": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of search queries"
                            }
                        },
                        "required": ["query_list"]
                    }
                }
            }
        """
        super().__init__(config, tool_schema)
        self.config = config.get("config_path")
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    def _connector(self, database: str) -> SQLiteConnector:
        return _connector(database, self.config)


    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
            tool_creation_response: The response of the tool when creating the instance.
        """
        if instance_id is None:
            instance_id = str(uuid.uuid4())
        self._instance_dict[instance_id] = {
            "extra_info": kwargs.get("extra_info") or {},
        }
        return instance_id, ToolResponse()



# class DatabaseTool(BaseVerlDBTool):
#     def execute(
#             self, instance_id: str, catalog: Optional[str] = "", include_sys: Optional[bool] = False
#     ) -> Tuple[str, float, dict]:
#         if self.agent_config.db_type == DBType.SQLITE:
#             result = FuncToolResult(result=[])
#         else:
#             result = self._tool_adapter("").list_databases(catalog, include_sys)
#         logger.debug(f"List database for {instance_id} result: {result}")
#         return json.dumps(result.model_dump(), ensure_ascii=False), 0.0, {}

# class SchemaTool(BaseVerlDBTool):
#     def execute(
#         self, instance_id: str, catalog: Optional[str] = "", database: Optional[str] = "", include_sys: bool = False
#     ) -> Tuple[str, float, dict]:
#         result = self._connector(database).get(catalog, database, include_sys)
#         if result.success:
#             logger.debug(f"List schemas for {instance_id} result: {len(result.result)}")
#         else:
#             logger.debug(f"List schemas for {instance_id} result: {result}")
#         return json.dumps(result.model_dump(), ensure_ascii=False), 0.0, {}


class ListTableTool(BaseVerlDBTool):
    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any], **kwargs
    ) -> Tuple[ToolResponse, float, dict]:
        try:
            include_views = parameters.get("include_views", False)
            database = self._instance_dict[instance_id]["extra_info"]["db_id"]
            connector = self._connector(database)
            tables = connector.get_tables("", database, "")
            if include_views:
                views = connector.get_views("", database, "")
                if views:
                    tables.extend(views)
            print(f"🔧🔧🔧[Tool-Call]List tables for {instance_id} result: {len(tables)}")
            result = FuncToolResult(result=tables)
        except Exception as e:
            result = FuncToolResult(success=0, error=str(e))
            print(f"🔧🔧🔧[Tool-Call]List tables for {instance_id} error: {e}")
        return ToolResponse(text=json.dumps(result.model_dump(), ensure_ascii=False)), 0.0, {}


class DescTableTool(BaseVerlDBTool):
    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any], **kwargs
    ) -> Tuple[ToolResponse, float, dict]:
        table_name = parameters.get("table_name")
        table_type = parameters.get("table_type", "table")
        try:
            database = self._instance_dict[instance_id]["extra_info"]["db_id"]
            connector = self._connector(database)
            if table_type == "table":
                exec_result = connector.get_tables_with_ddl(
                    "", database, "", tables=[table_name]
                )
                result = FuncToolResult(result=exec_result)
            elif table_type == "view":
                exec_result = connector.get_views_with_ddl(
                    "", database, "", tables=[table_name]
                )
                result = FuncToolResult(result=exec_result)
            else:
                result = FuncToolResult(success=0, error=f"Unknown table type: {table_type}")
        except Exception as e:
            result = FuncToolResult(success=0, error=f"Error describing table {table_name}: {e}")
        print(f"🔧🔧🔧[Too-Call]：Describe tables {table_name} for {instance_id} result: {result}")

        return ToolResponse(text=json.dumps(result.model_dump(), ensure_ascii=False)), 0.0, {}

from verl.tools.compress_util import DataCompressor

class QueryTool(BaseVerlDBTool):
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[ToolResponse, float, dict]:
        try:
            sql = parameters.get("sql")
            database = self._instance_dict[instance_id]["extra_info"]["db_id"]
            result = self._connector(database).execute_query(sql)
            if result.success:
                result = DataCompressor.quick_compress(data = result.sql_return)
                print(f"🔧🔧🔧[Too-Call]：Execute query for {instance_id} result: {result}")
            else:
                print(f"⚠️⚠️⚠[Too-Call]：Execute query for {instance_id} failed: {result.error}")
        except Exception as e:
            result = FuncToolResult(success=0, error=str(e))
        return ToolResponse(text=json.dumps(result.model_dump(), ensure_ascii=False)), 0.0, {}

