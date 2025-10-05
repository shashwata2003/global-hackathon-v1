from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from state import DataPipelineState
import pandas as pd
import json
from dotenv import load_dotenv
import os
from typing import Optional

# --------------------------
# Load Environment Variables
# --------------------------
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("⚠️ GEMINI_API_KEY not found in .env or environment variables.")

# --------------------------
# Initialize Gemini LLM via LangChain
# --------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # use "gemini-1.5-pro" if you want deeper reasoning
    temperature=0.2,
    api_key=api_key
)

# --------------------------
# Prompt Template
# --------------------------
insight_prompt = ChatPromptTemplate.from_template("""
You are a data visualization and insights expert.
You are given a sample of queried data from a CSV file.
Your task is to:
1. Suggest the most appropriate type(s) of chart(s) (bar, line, pie, scatter, histogram, etc.).
2. Identify which columns should be used for each chart.
3. Write 2-3 concise business insights based on the data.

Respond in **valid JSON** format exactly like this:
{{
  "charts": [
    {{
      "type": "<chart_type>",
      "x": "<x_column>",
      "y": "<y_column>",
      "title": "<chart_title>"
    }}
  ],
  "insights": [
    "<insight_1>",
    "<insight_2>"
  ]
}}

Data Sample:
{data_sample}

Metadata:
{metadata}

User Query:
{user_query}
""")

# --------------------------
# Output Agent Function
# --------------------------
def run_output_agent(state: DataPipelineState) -> DataPipelineState:
    """
    Generates visualization suggestions and business insights 
    using queried data and metadata.
    """
    if state.queried_output is None or state.queried_output.empty:
        raise ValueError("No data available for output generation.")

    # Get a small representative data sample
    data_sample = state.queried_output.head(5).to_dict(orient="records")

    # Build chain (Prompt → Gemini LLM)
    chain = insight_prompt | llm

    # Run LLM
    response = chain.invoke({
        "data_sample": json.dumps(data_sample, indent=2),
        "metadata": json.dumps(state.metadata, indent=2),
        "user_query": state.user_query
    })

    # Parse JSON safely
    try:
        parsed_output = json.loads(response.content)
    except Exception:
        parsed_output = {
            "charts": [],
            "insights": [response.content.strip()]
        }

    state.output = parsed_output
    print("\n🧠 Output Agent Suggestions:\n", json.dumps(parsed_output, indent=2))
    return state
