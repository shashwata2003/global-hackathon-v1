from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from state import DataPipelineState
import pandas as pd
import json

# LLM Initialization (Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

# Prompt for generating visualization insights
insight_prompt = ChatPromptTemplate.from_template("""
You are a data visualization and insights expert. 
You are given a sample of queried data from a CSV file. 
Your task is to:
1. Suggest the most appropriate type(s) of chart(s) (like bar, line, pie, scatter, histogram, etc.) to visualize the data.
2. Identify which columns should be used for each chart.
3. Write 2-3 clear and concise insights or observations a business person can understand from this data.

Respond in **valid JSON** format as:
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

def run_output_agent(state: DataPipelineState) -> DataPipelineState:
    """
    Generates insights and chart instructions from the queried data.
    """

    if state.queried_output is None or state.queried_output.empty:
        raise ValueError("No data available for output generation.")

    # Get a small data sample for the LLM (to avoid token overload)
    data_sample = state.queried_output.head(5).to_dict(orient="records")

    # Create the prompt
    chain = insight_prompt | llm

    response = chain.invoke({
        "data_sample": json.dumps(data_sample, indent=2),
        "metadata": json.dumps(state.metadata, indent=2),
        "user_query": state.user_query
    })

    try:
        parsed_output = json.loads(response.content)
    except Exception:
        # fallback for non-JSON responses
        parsed_output = {
            "charts": [],
            "insights": [response.content.strip()]
        }

    state.output = parsed_output
    print("\n🧠 Output Agent Suggestions:\n", json.dumps(parsed_output, indent=2))
    return state
