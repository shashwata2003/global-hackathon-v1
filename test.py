import pandas as pd
from langgraph.graph import StateGraph, END
from agents.planner_agent import planner_node
from agents.sql_agent import sql_agent_node
from agents.validator_agent import validator_agent_node
from agents.output_agent import run_output_agent
from generatemetadata import generate_metadata_from_df
from state import DataPipelineState


# -----------------------
# Build LangGraph
# -----------------------
def build_graph():
    graph = StateGraph(DataPipelineState)

    # Add nodes
    graph.add_node("planner_agent", planner_node)
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("validator_agent", validator_agent_node)
    graph.add_node("output_agent", run_output_agent)

    # Define edges
    graph.add_edge("planner_agent", "sql_agent")
    graph.add_edge("sql_agent", "validator_agent")

    # Conditional validation edge
    def check_validation(state: DataPipelineState):
        if hasattr(state, "validation_result") and state.validation_result and state.validation_result.lower() == "valid":
            return "output_agent"
        else:
            print("🔁 Validation failed. Re-running pipeline...")
            return "planner_agent"

    graph.add_conditional_edges("validator_agent", check_validation)

    graph.set_entry_point("planner_agent")
    graph.add_edge("output_agent", END)

    return graph.compile()


# -----------------------
# Test Script
# -----------------------
if __name__ == "__main__":
    # Load test CSV
    df = pd.read_csv("transactions_raw.csv")  # replace with a small test CSV
    metadata = generate_metadata_from_df(df)

    # Initialize pipeline state
    state = DataPipelineState(
        metadata=metadata,
        user_query="Show me total sales by region for the last quarter",
        df=df,
        queried_output=None,
    )

    # Build and run the graph
    app = build_graph()
    final_state = app.invoke(state)

    print("\n✅ Pipeline Execution Complete!\n")
    print("🧩 Final Output:")
    print(state.output)

    # Optional: print logs
    print("\n📜 Logs:")
    for log in final_state.logs:
        print(log)
