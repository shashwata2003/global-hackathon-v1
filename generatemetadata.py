import os
import pandas as pd
import json
import google.generativeai as genai
from dotenv import load_dotenv

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-2.5-flash"  # Fast + cost-efficient for metadata generation


# ----------------------------
# Generate Metadata from DataFrame
# ----------------------------
def generate_metadata_from_df(df: pd.DataFrame):
    """
    Generates AI-based metadata for a given DataFrame using Gemini.
    
    Args:
        df (pd.DataFrame): The DataFrame containing raw business data.
        
    Returns:
        dict: Metadata JSON describing each column.
    """
    if df is None or df.empty:
        raise ValueError("❌ DataFrame is empty or None.")

    # Summarize columns
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

    # Build prompt
    prompt = f"""
You are a data analyst AI.
Given the following column info from a business transaction dataset, 
return a JSON metadata schema where each column has:
- "column_name"
- "data_type" (based on values)
- "description" (1-2 lines about what it likely represents)

Columns summary:
{json.dumps(columns_summary, indent=2)}

Respond ONLY with valid JSON (no explanations, no markdown, no ```json).
    """

    # Call Gemini
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    text = response.text.strip()

    # Try parsing JSON safely
    try:
        metadata = json.loads(text)
    except json.JSONDecodeError:
        print("⚠️ Model output not valid JSON, returning raw text instead.")
        metadata = {"raw_text": text}

    return metadata


# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    # Simulate DataFrame (for testing)
    data = {
        "account_id": [101, 102, 103, 104],
        "transaction_amount": [200.5, 340.0, 120.75, 560.9],
        "payment_method": ["PayPal", "Card", "UPI", "Wallet"],
        "transaction_date": ["2025-09-10", "2025-09-12", "2025-09-13", "2025-09-14"]
    }

    df = pd.DataFrame(data)
    metadata = generate_metadata_from_df(df)
    print(json.dumps(metadata, indent=2))
