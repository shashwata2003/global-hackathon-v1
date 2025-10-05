import os
import sys
sys.path.append(os.path.dirname(__file__))

import streamlit as st
import pandas as pd
# import plotly.express as px
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

uploaded_file = st.file_uploader("📂 Upload CSV/Excel file", type=["csv", "xlsx"])

if uploaded_file:
    # Load file
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ File uploaded successfully!")

        # Display sample & summary
        st.write("### 📄 Preview of Data")
        st.dataframe(df.head())

        st.write("### ℹ️ Dataset Summary")
        st.write(f"- Number of rows: {df.shape[0]}")
        st.write(f"- Number of columns: {df.shape[1]}")
        st.write(f"- Columns: {', '.join(df.columns)}")

        # If query is provided
        user_query = st.text_input("💬 Enter your query:", placeholder="e.g., Show me total sales by region for the last quarter")
        if user_query:
            metadata = generate_metadata_from_df(df)
            state = DataPipelineState(
                metadata=metadata,
                user_query=user_query,
                df=df,
                queried_output=None,
                output=None
            )

        if st.button("🚀 Run Pipeline"):
            with st.spinner("Running multi-agent pipeline..."):
                app = build_graph()
                final_state = app.invoke(state)

            st.success("✅ Pipeline Execution Complete!")

            # Debug: show raw output
            st.write("### 🔹 Raw Pipeline Output")
            st.write(final_state["output"]["charts"])

            # Display insights
            if final_state.get("output") and final_state["output"]:
                insights = final_state["output"].get("insights", [])
                if insights:
                    st.write("### 💡 Insights")
                    for insight in insights:
                        st.write(f"- {insight}")

                # # Display charts
                # charts = final_state["output"].get("charts", [])
                # if charts:
                #     st.write("### 📊 Charts")
                #     for chart in charts:
                #         chart_type = chart.get("type")
                #         x_col = chart.get("x")
                #         y_col = chart.get("y")
                #         title = chart.get("title", "")

                #         plot_df = getattr(final_state, "queried_data", df)

                #         if chart_type == "bar":
                #             fig = px.bar(plot_df, x=x_col, y=y_col, title=title)
                #             st.plotly_chart(fig, use_container_width=True)
                #         elif chart_type == "line":
                #             fig = px.line(plot_df, x=x_col, y=y_col, title=title)
                #             st.plotly_chart(fig, use_container_width=True)
                #         elif chart_type == "scatter":
                #             fig = px.scatter(plot_df, x=x_col, y=y_col, title=title)
                #             st.plotly_chart(fig, use_container_width=True)


    except Exception as e:
        st.error(f"❌ Error loading file: {e}")

else:
    st.info("📥 Please upload a CSV or Excel file to begin.")
