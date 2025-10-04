import sqlite3
import pandas as pd
from state import DataPipelineState

# ----------------------------
# Convert DataFrame to SQLite
# ----------------------------
def df_to_sqlite(df: pd.DataFrame, db_path: str = "temp_db.sqlite", table_name: str = "data"):
    """
    Load a pandas DataFrame into SQLite.
    Overwrites existing database file.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")

    # Overwrite DB if exists
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    return db_path, table_name

# ----------------------------
# Generate SQL from Planner output
# ----------------------------
def generate_sql_from_plan(plan: dict, table_name: str) -> str:
    if not plan:
        raise ValueError("Planner plan is empty.")

    # SELECT columns
    columns = plan.get("columns_to_use", ["*"])
    select_clause = ", ".join(columns)

    # Aggregations
    aggs = plan.get("aggregations", [])
    agg_clauses = []
    for agg in aggs:
        col = agg.get("column")
        typ = agg.get("type")
        alias = agg.get("alias", f"{typ}_{col}")
        if typ == "sum":
            agg_clauses.append(f"SUM({col}) AS {alias}")
        elif typ == "avg":
            agg_clauses.append(f"AVG({col}) AS {alias}")
        elif typ == "count":
            agg_clauses.append(f"COUNT({col}) AS {alias}")
        elif typ == "min":
            agg_clauses.append(f"MIN({col}) AS {alias}")
        elif typ == "max":
            agg_clauses.append(f"MAX({col}) AS {alias}")
        elif typ == "pct_change":
            agg_clauses.append(f"{col} AS {alias}")

    if agg_clauses:
        select_clause += ", " + ", ".join(agg_clauses)

    # FROM
    sql = f"SELECT {select_clause} FROM {table_name}"

    # WHERE filters
    filters = plan.get("filters", [])
    where_clauses = []
    for f in filters:
        col = f.get("column")
        op = f.get("operator", "=")
        val = f.get("value")
        if isinstance(val, str):
            val = f"'{val}'"
        elif isinstance(val, list):
            if op.lower() == "between" and len(val) == 2:
                val = f"{val[0]} AND {val[1]}"
                op = "BETWEEN"
            else:
                val = "(" + ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in val]) + ")"
        where_clauses.append(f"{col} {op} {val}")

    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)

    # GROUP BY
    group_by = plan.get("group_by", [])
    if group_by:
        sql += " GROUP BY " + ", ".join(group_by)

    # ORDER BY
    order_by = plan.get("order_by", [])
    if order_by:
        order_clauses = [f"{o['column']} {o['direction']}" for o in order_by]
        sql += " ORDER BY " + ", ".join(order_clauses)

    # LIMIT
    limit = plan.get("limit")
    if limit:
        sql += f" LIMIT {limit}"

    return sql + ";"

# ----------------------------
# SQL Agent Node
# ----------------------------
def sql_agent_node(state: DataPipelineState, db_path: str = "temp_db.sqlite") -> DataPipelineState:
    """
    LangGraph node: executes SQL generated from Planner plan using a DataFrame stored in state.df.
    """
    try:
        if not hasattr(state, "df") or state.df is None:
            raise ValueError("State has no DataFrame (state.df) to query.")

        # Convert DataFrame to SQLite
        db_path, table_name = df_to_sqlite(state.df, db_path=db_path)
        state.add_log(f"SQL Agent: DataFrame loaded into {db_path} as table {table_name}")

        # Generate SQL
        sql_query = generate_sql_from_plan(state.plan, table_name)
        state.sql_query = sql_query
        state.add_log(f"SQL Agent: Generated SQL query:\n{sql_query}")

        # Execute SQL
        conn = sqlite3.connect(db_path)
        df_result = pd.read_sql_query(sql_query, conn)
        conn.close()

        state.queried_data = df_result
        state.add_log(f"SQL Agent: Query executed successfully. Retrieved {len(df_result)} rows.")
        return state

    except Exception as e:
        state.add_log(f"SQL Agent Error: {str(e)}")
        state.queried_data = None
        return state
