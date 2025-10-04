import os
import pandas as pd
import numpy as np
import google.generativeai as genai
import json
from dotenv import load_dotenv

load_dotenv()
# ---------- CONFIG ----------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "gemini-2.5-flash"  # fast + cheap for metadata generation

# ---------- FUNCTION TO GENERATE METADATA ----------
def generate_metadata(csv_path):
    df = pd.read_csv(csv_path)

    columns_summary = []
    for col in df.columns:
        sample_values = df[col].dropna().unique().tolist()[:5]
        inferred_dtype = str(df[col].dtype)
        summary = {
            "column_name": col,
            "data_type": inferred_dtype,
            "sample_values": sample_values
        }
        columns_summary.append(summary)

    prompt = f"""
You are a data analyst AI.
Given the following column info from a business transaction dataset, 
return a JSON metadata schema where each column has:
- "column_name"
- "data_type" (based on values)
- "description" (1-2 lines about what it likely represents)

Columns summary:
{json.dumps(columns_summary, indent=2)}

Respond ONLY with valid JSON. Without backticks or and starting strings like ```json
    """

    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    text = response.text.strip()

    # Attempt to parse JSON safely
    try:
        metadata = json.loads(text)
    except Exception:
        print("⚠️ Model output not valid JSON, returning raw text instead.")
        metadata = text

    return metadata


# # ---------- MAIN EXECUTION ----------
# if __name__ == "__main__":
#     csv_path = "transactions_raw.csv"  # use your generated CSV
#     metadata = generate_metadata(csv_path)

#     # Save metadata to JSON file
#     with open("metadata.json", "w", encoding="utf-8") as f:
#         json.dump(metadata, f, indent=2, ensure_ascii=False)

#     print("✅ Metadata generated and saved to metadata.json")
