# planner_agent.py
import os
import json
import uuid
from typing import Any, Dict, List, Optional

# Google Gen AI (Gemini) client
# using the new google-genai SDK (recommended)
from google import genai
from state import DataPipelineState
from dotenv import load_dotenv

load_dotenv()


# For LangGraph node
try:
    from langgraph.graph import StateGraph, START, END
except Exception:
    # If LangGraph not installed or you're running just the helper, this is fine.
    StateGraph = None
    START = None
    END = None

# --------------------------
# CONFIG
# --------------------------
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # change if you want
API_KEY_ENV = os.getenv("GEMINI_API_KEY")  # set this in your environment

# Initialize client (will use API key from env if provided)
def _init_genai_client(api_key: Optional[str] = None):
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment. Set GEMINI_API_KEY.")
    client = genai.Client(api_key=api_key)
    return client

# --------------------------
# Helper: compact metadata string for prompt
# --------------------------
def _build_metadata_summary(metadata: List[Dict[str, Any]], max_samples: int = 5) -> str:
    """
    Convert metadata (list of {column_name, data_type, description?, sample_values}) into
    a short text that we can put in the prompt.
    """
    lines = []
    for col in metadata:
        name = col.get("column_name")
        dtype = col.get("data_type", "unknown")
        desc = col.get("description", "")
        samples = col.get("sample_values", [])[:max_samples]
        lines.append({
            "column_name": name,
            "data_type": dtype,
            "description": desc,
            "sample_values": samples
        })
    # keep it JSON-ish but small
    return json.dumps(lines, indent=2, ensure_ascii=False)

# --------------------------
# Main: call Gemini to produce a plan
# --------------------------
def generate_plan_with_gemini(
    metadata: List[Dict[str, Any]],
    user_query: str,
    model_name: str = MODEL_NAME,
    api_key: Optional[str] = None,
    max_retries: int = 1
) -> Dict[str, Any]:
    """
    Ask Gemini to produce a structured "execution plan" (JSON) based on metadata + user query.
    Returns parsed JSON dict (plan). If Gemini returns non-JSON, we attempt a safe fallback.
    """
    client = _init_genai_client(api_key)

    metadata_summary = _build_metadata_summary(metadata)

    prompt = f"""
You are a Planner AI for a data-analysis pipeline. Input: a dataset metadata description
and a user's natural-language question. Your job is to produce a JSON "plan" the downstream
agents can follow to satisfy the user query.

Constraints:
- Respond ONLY with valid JSON (no extra text).
- Use the column names EXACTLY as provided in the metadata.
- For "filters" prefer conservative suggestions (e.g., date ranges, categories).
- Provide "hints" to downstream agents about parsing/casting issues (e.g., 'cast Amount to numeric, handle "N/A"').
- Include "confidence" (0.0-1.0). If uncertain, set confidence < 0.6.

Required JSON schema (produce these keys):
{{
  "plan_id": "<uuid>",
  "columns_to_use": [ "<col1>", "<col2>", ... ],
  "filters": [
     {{ "column": "<col>", "operator": "<=|>=|=|between|in|like|contains>", "value": "<value or [from,to]>", "reason": "<short reason>" }}
  ],
  "aggregations": [
     {{ "type": "sum|avg|count|min|max|pct_change", "column": "<col>", "alias": "<alias>" }}
  ],
  "group_by": [ "<colA>", ... ],
  "order_by": [ {{ "column": "<col_or_alias>", "direction": "asc|desc" }} ],
  "limit": <integer or null>,
  "sql_template": "<optional - SQL-like skeleton using {{table}} placeholder>",
  "hints": [ "<hint1>", ... ],
  "steps": [ "<short step 1>", "<short step 2>", ... ],
  "confidence": <float 0.0 - 1.0>,
  "explain": "<1-2 sentence human-readable explanation of plan>"
}}

Input metadata (trimmed):
{metadata_summary}

User question:
\"\"\"{user_query}\"\"\"

Produce the JSON plan now.
    """

    # call the model
    response = client.models.generate_content(model=model_name, contents=prompt)
    text = getattr(response, "text", None) or (response.text() if callable(getattr(response, "text", None)) else None)
    if text is None:
        # best-effort: str(response)
        text = str(response)

    text = text.strip()

    # Attempt parse
    try:
        plan = json.loads(text)
        return plan
    except Exception:
        # fallback: try to extract first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            maybe = text[start:end+1]
            try:
                plan = json.loads(maybe)
                return plan
            except Exception:
                pass

    # If parsing fails, return a conservative plan and include raw model output for debugging.
    fallback_plan = {
        "plan_id": str(uuid.uuid4()),
        "columns_to_use": [col.get("column_name") for col in metadata[:3]],
        "filters": [],
        "aggregations": [],
        "group_by": [],
        "order_by": [],
        "limit": None,
        "sql_template": None,
        "hints": ["Model returned non-JSON; inspect 'raw_model_output' for details"],
        "steps": ["inspect model output", "ask SME agent for clarification"],
        "confidence": 0.25,
        "explain": "Fallback plan because the LLM output could not be parsed as JSON.",
        "raw_model_output": text
    }
    return fallback_plan

# --------------------------
# LangGraph node wrapper
# --------------------------
def planner_node(state: DataPipelineState, config: Any = None, runtime: Any = None) -> DataPipelineState:
    """
    LangGraph-style node for the Planner Agent.
    Uses the DataPipelineState for input/output.
    """
    if not state.metadata or not state.user_query:
        state.add_log("Planner: missing metadata or user_query.")
        return state

    try:
        plan = generate_plan_with_gemini(state.metadata, state.user_query)
        state.plan = plan  
        state.plan_steps = plan.get("steps", [])
        state.add_log(f"Planner: Generated plan with confidence {plan.get('confidence', 0)}.")
        # The planner determines next node
        state.add_log("Planner: Passing control to SME Agent.")
        return state
    except Exception as e:
        state.add_log(f"Planner error: {str(e)}")
        state.plan = None
        return state


# # --------------------------
# # Quick example: run planner standalone
# # --------------------------
# if __name__ == "__main__":
#     # Very small mock metadata (in your system you'd load metadata.json)
#     sample_metadata = [
#         {"column_name": "TransactionID", "data_type": "string", "description": "Transaction unique ID", "sample_values": ["a1","b2","c3"]},
#         {"column_name": "Sender", "data_type": "string", "description": "Sender name or account", "sample_values": ["Alice","Bob"]},
#         {"column_name": "Receiver", "data_type": "string", "description": "Receiver name or account", "sample_values": ["Charlie","David"]},
#         {"column_name": "Amount", "data_type": "float_or_string", "description": "Transaction amount (may include 'N/A')", "sample_values": ["123.45","N/A","???" ]},
#         {"column_name": "Timestamp", "data_type": "string", "description": "Event timestamp (various formats)", "sample_values": ["2024-04-12","12/04/2024","03-05-2025 14:55:23"]}
#     ]

#     # small test question
#     q = "Why did our sales dip in September 2025? Show top drivers by product and country."

#     # ensure env var is set (or set before running)
#     if not os.getenv("GEMINI_API_KEY"):
#         print("Set GEMINI_API_KEY environment variable before running this example.")
#     else:
#         plan = generate_plan_with_gemini(sample_metadata, q)
#         print(json.dumps(plan, indent=2, ensure_ascii=False))
