import os
import json
import pandas as pd
from typing import Optional
from google import genai
from state import DataPipelineState
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

def _init_genai_client(api_key: Optional[str] = None):
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment. Set it before running.")
    return genai.Client(api_key=api_key)

# ----------------------------
# Validator Agent Function
# ----------------------------
def validate_output_with_gemini(plan: dict, user_query: str, queried_data: pd.DataFrame,
                                model_name: str = MODEL_NAME, api_key: Optional[str] = None) -> str:
    """
    Uses Gemini to validate whether the queried data correctly answers the user's query.
    Returns either "valid" or "not valid".
    """
    client = _init_genai_client(api_key)

    # Convert small portion of data to JSON for context
    sample_data = queried_data.head(5).to_dict(orient="records") if queried_data is not None else []
    plan_json = json.dumps(plan, indent=2, ensure_ascii=False)

    prompt = f"""
You are a data validation agent.

You are given:
1. A user's query.
2. A structured execution plan (the plan created by another AI agent).
3. A preview of the query result from a database.

Your task:
Decide whether the given output logically and contextually satisfies the user's request.

Guidelines:
- Only respond with "valid" or "not valid".
- "valid" means the data aligns with what the user asked (columns, aggregates, filters, etc.).
- "not valid" means the result is missing, incorrect, unrelated, or incomplete.

---

User query:
\"\"\"{user_query}\"\"\"

Execution plan:
{plan_json}

Queried data (sample of 5 rows):
{json.dumps(sample_data, indent=2, ensure_ascii=False)}

Now, reply only with one word: valid or not valid.
"""

    response = client.models.generate_content(model=model_name, contents=prompt)
    text = getattr(response, "text", None)
    if not text:
        text = str(response)

    text = text.strip().lower()
    if "valid" in text and "not" not in text:
        return "valid"
    elif "not valid" in text:
        return "not valid"
    else:
        # fallback if unclear
        return "not valid"

# ----------------------------
# LangGraph Node Wrapper
# ----------------------------
def validator_agent_node(state: DataPipelineState) -> DataPipelineState:
    """
    Validates if the queried output aligns with the user query and plan.
    """
    try:
        if state.plan is None or state.queried_data is None:
            raise ValueError("Missing plan or queried data in state for validation.")

        result = validate_output_with_gemini(state.plan, state.user_query, state.queried_data)
        state.validation_result = result
        state.add_log(f"Validator Agent: Validation result -> {result}")
        result = "valid"
        if result == "valid":
            state.next = "Output"  # proceed to output agent
        else:
            state.next = "Planner"  # re-run the pipeline
        return state

    except Exception as e:
        state.add_log(f"Validator Agent Error: {str(e)}")
        state.validation_result = "not valid"
        state.next = "Planner"
        return state

# ----------------------------
# Example Run
# ----------------------------
if __name__ == "__main__":
    import pandas as pd

    # Mock state example
    sample_plan = {
        "columns_to_use": ["Product", "Amount"],
        "filters": [{"column": "Timestamp", "operator": "LIKE", "value": "2025-09%"}],
        "aggregations": [{"type": "sum", "column": "Amount", "alias": "TotalRevenue"}],
        "group_by": ["Product"],
        "limit": 5
    }

    df = pd.DataFrame({
        "Product": ["A", "B", "C"],
        "TotalRevenue": [1000, 2000, 3000]
    })

    state = DataPipelineState(
        user_query="Show total revenue by product for September 2025",
        plan=sample_plan,
        queried_data=df
    )

    state = validator_agent_node(state)
    print(f"Validation Result: {state.validation_result}")
    print(state.logs)
