import sqlite3
import pandas as pd
import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from state import DataPipelineState


# ----------------------------
# Convert DataFrame to SQLite
# ----------------------------
def df_to_sqlite(df: pd.DataFrame, db_path: str = "temp_db.sqlite", table_name: str = "data"):
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    return db_path, table_name


# ----------------------------
# Generate SQL using LLM
# ----------------------------
def generate_sql_with_llm(plan: dict, df: pd.DataFrame, model_name: str = "gemini-1.5-pro") -> str:
    """
    Use an LLM to generate SQL from the planner plan and DataFrame schema.
    """
    if not plan:
        raise ValueError("Planner plan is empty.")

    schema_info = {
        "columns": list(df.columns),
        "sample_rows": df.head(3).to_dict(orient="records")
    }

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful SQL generation assistant. Given a data table and a user plan,
    write a valid SQLite SQL query that fulfills the plan.

    Table schema:
    {schema}

    Planner plan (JSON):
    {plan}

    Rules:
    - Use only valid SQLite syntax.
    - Assume the table name is "data".
    - Always output *only* the SQL query, nothing else.
    """)

    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    chain = prompt | llm

    sql_query = chain.invoke({
        "schema": json.dumps(schema_info, indent=2),
        "plan": json.dumps(plan, indent=2)
    }).content.strip()

    return sql_query


# ----------------------------
# SQL Agent Node
# ----------------------------
def sql_agent_node(state: DataPipelineState, db_path: str = "temp_db.sqlite") -> DataPipelineState:
    """
    LangGraph node: generates SQL via LLM and executes it on the SQLite DB.
    """
    try:
        if not hasattr(state, "df") or state.df is None:
            raise ValueError("State has no DataFrame (state.df) to query.")
        if not hasattr(state, "plan") or state.plan is None:
            raise ValueError("Planner output (state.plan) is missing.")

        # Convert DataFrame to SQLite
        db_path, table_name = df_to_sqlite(state.df, db_path=db_path)
        state.add_log(f"SQL Agent: DataFrame loaded into {db_path} as table {table_name}")

        # Generate SQL using LLM
        sql_query = generate_sql_with_llm(state.plan, state.df)
        state.sql_query = sql_query
        state.add_log(f"SQL Agent: Generated SQL query via LLM:\n{sql_query}")

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
