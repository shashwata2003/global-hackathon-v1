from typing import Optional, Any, Dict, List
from pydantic import BaseModel
import pandas as pd

class DataPipelineState(BaseModel):
    """
    Shared state for all agents in the Autonomous Insight Pipeline.
    Each agent will read/write relevant attributes as the graph progresses.
    """

    # --- Core Input ---
    user_query: str
    metadata: Optional[List[Dict[str, Any]]] = None
    df: Optional[pd.DataFrame] = None  # Make optional

    # --- Planner Agent Output ---
    plan: Dict[str, Any] = {}
    plan_steps: Optional[List[str]] = None

    # --- SME Agent Output ---
    sme_output: Optional[Dict[str, Any]] = None

    # --- SQL Agent Output ---
    sql_query: Optional[str] = None
    queried_data: Optional[Any] = None

    # --- Validator Agent Output ---
    validation_result: Optional[str] = None
    validation_status: Optional[bool] = None
    validation_feedback: Optional[str] = None

    # --- Output Agent Output ---
    insights: Optional[str] = None
    visuals: Optional[List[str]] = None
    next: str = ""
    output: Optional[Dict[str, Any]] = None

    # --- System Logs ---
    logs: List[str] = []

    # -------------------------
    # Helper Methods
    # -------------------------
    def add_log(self, message: str):
        self.logs.append(message)

    def to_dict(self) -> Dict[str, Any]:
        return self.dict()

    # -------------------------
    # Pydantic v2 Config
    # -------------------------
    model_config = {
        "arbitrary_types_allowed": True
    }
