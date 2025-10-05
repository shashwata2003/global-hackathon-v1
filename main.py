import streamlit as st
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
        if hasattr(state, "validation_result") and state.validation_result.lower() == "valid":
            return "output_agent"
        else:
            st.warning("🔁 Validation failed. Re-running pipeline...")
            return "planner_agent"

    graph.add_conditional_edges("validator_agent", check_validation)

    graph.set_entry_point("planner_agent")
    graph.add_edge("output_agent", END)

    return graph.compile()


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Autonomous Insight Pipeline", layout="wide")
st.title("🧠 Autonomous Insight Pipeline")

st.write(
    """
    Upload a dataset and enter a query to run the multi-agent data pipeline 
    (Planner → SQL → Validator → Output).  
    """
)

uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])
user_query = st.text_input("💬 Enter your query:", placeholder="e.g., Show me total sales by region for the last quarter")

if uploaded_file and user_query:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.write("### Preview of Data")
        st.dataframe(df.head())

        # Generate metadata
        metadata = generate_metadata_from_df(df)

        # Initialize state
        state = DataPipelineState(
            metadata=metadata,
            user_query=user_query,
            df = df,
            queried_output=None,
            output=None
        )

        if st.button("🚀 Run Pipeline"):
            with st.spinner("Running multi-agent pipeline..."):
                app = build_graph()
                final_state = app.invoke(state)

            st.success("✅ Pipeline Execution Complete!")
            st.write("### 🧩 Final Output:")
            if isinstance(final_state.output, pd.DataFrame):
                st.dataframe(final_state.output)
            else:
                st.write(final_state.output)

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info("📥 Please upload a CSV file and enter a query to begin.")
