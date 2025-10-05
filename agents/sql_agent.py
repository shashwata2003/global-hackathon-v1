import sqlite3
import pandas as pd
import os
import json
from state import DataPipelineState
from dotenv import load_dotenv
from google import genai
from typing import Optional

load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _init_genai_client(api_key: Optional[str] = None):
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment. Set it before running.")
    return genai.Client(api_key=api_key)


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
def generate_sql_with_llm(plan: dict, df: pd.DataFrame, model_name: str = MODEL_NAME) -> str:
    if not plan:
        raise ValueError("Planner plan is empty.")

    schema_info = {
        "columns": list(df.columns),
        "sample_rows": df.head(3).to_dict(orient="records")
    }

    prompt = f"""
You are a helpful SQL generation assistant. Given a data table and a user plan,
write a valid SQLite SQL query that fulfills the plan.

Table schema:
{json.dumps(schema_info, indent=2)}

Planner plan (JSON):
{json.dumps(plan, indent=2)}

Rules:
- Use only valid SQLite syntax.
- Assume the table name is "data".
- Always output *only* the SQL query, nothing else.
- You may reference derived columns suggested in 'derived_columns'.

"""

    client = _init_genai_client()
    response = client.models.generate_content(model=model_name, contents=[prompt])

    # Extract raw SQL
    raw_sql = response.candidates[0].content.parts[0].text

    # Remove ```sqlite or ``` if present
    sql_query = raw_sql.strip().replace("```sqlite", "").replace("```", "").strip()

    return sql_query




# ----------------------------
# SQL Agent Node
# ----------------------------
def sql_agent_node(state: DataPipelineState, db_path: str = "temp_db.sqlite") -> DataPipelineState:
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


# # ----------------------------
# # Main function to test
# # ----------------------------
# def main():
#     # Sample DataFrame
#     df = pd.DataFrame({
#         "region": ["North", "South", "East", "West", "North"],
#         "sales": [100, 150, 200, 130, 170],
#         "quarter": ["Q1", "Q1", "Q1", "Q1", "Q2"]
#     })

#     # Sample Planner Plan
#     plan = {
#         "goal": "Calculate total sales by region for the last quarter",
#         "aggregations": [{"column": "sales", "operation": "sum"}],
#         "group_by": ["region"]
#     }

#     # Initialize State
#     state = DataPipelineState(
#         user_query="Show total sales by region for the last quarter",
#         df=df,
#         plan=plan
#     )

#     # Run SQL Agent
#     state = sql_agent_node(state)

#     # Print Results
#     print("\n=== Logs ===")
#     for log in state.logs:
#         print(log)

#     print("\n=== Generated SQL Query ===")
#     print(state.sql_query)

#     print("\n=== Query Results ===")
#     if state.queried_data is not None:
#         print(state.queried_data)
#     else:
#         print("No data returned from query.")


# if __name__ == "__main__":
#     main()
