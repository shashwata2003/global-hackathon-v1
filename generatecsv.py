import pandas as pd
import random
import uuid
import numpy as np
from datetime import datetime, timedelta

# ---------- CONFIG ----------
num_rows = 10000
output_file = "transactions_raw.csv"

# ---------- SAMPLE DATA ----------
names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Hannah", "Ivan", "Jack"]
countries = ["USA", "India", "UK", "Germany", "France", "Canada", "Australia"]
currencies = ["USD", "INR", "EUR"]
payment_methods = ["Credit Card", "Debit Card", "PayPal Balance", "Bank Transfer", "Crypto"]
statuses = ["Completed", "Pending", "Failed", "Reversed"]

# ---------- HELPER FUNCTIONS ----------
def random_date():
    start = datetime(2022, 1, 1)
    end = datetime(2025, 10, 1)
    delta = end - start
    random_days = random.randint(0, delta.days)
    dt = start + timedelta(days=random_days, seconds=random.randint(0, 86400))
    # introduce inconsistent date formats
    if random.random() < 0.3:
        return dt.strftime("%d/%m/%Y")
    elif random.random() < 0.6:
        return dt.strftime("%Y-%m-%d")
    else:
        return dt.strftime("%m-%d-%Y %H:%M:%S")

def maybe_missing(value):
    if random.random() < 0.03:
        return ""
    elif random.random() < 0.03:
        return None
    return value

def random_amount():
    amt = round(random.uniform(1.0, 5000.0), 2)
    # add some noise
    if random.random() < 0.01:
        return "N/A"
    elif random.random() < 0.01:
        return "???"
    return amt

# ---------- DATA GENERATION ----------
data = []
for _ in range(num_rows):
    transaction_id = str(uuid.uuid4())
    sender = random.choice(names)
    receiver = random.choice([n for n in names if n != sender])
    country = random.choice(countries)
    currency = random.choice(currencies)
    amount = random_amount()
    date = random_date()
    status = random.choice(statuses)
    payment_method = random.choice(payment_methods)
    
    # introduce some random casing issues
    sender = sender.lower() if random.random() < 0.2 else sender.upper() if random.random() < 0.1 else sender
    receiver = receiver.title() if random.random() < 0.1 else receiver
    
    # maybe introduce duplicates or errors
    if random.random() < 0.01:
        transaction_id = transaction_id[:8]  # truncated id
    
    row = {
        "TransactionID": maybe_missing(transaction_id),
        "Sender": maybe_missing(sender),
        "Receiver": maybe_missing(receiver),
        "Country": maybe_missing(country),
        "Currency": maybe_missing(currency),
        "Amount": maybe_missing(amount),
        "PaymentMethod": maybe_missing(payment_method),
        "Status": maybe_missing(status),
        "Timestamp": maybe_missing(date)
    }
    data.append(row)

df = pd.DataFrame(data)

# shuffle rows a bit
df = df.sample(frac=1).reset_index(drop=True)

# ---------- SAVE ----------
df.to_csv(output_file, index=False, encoding="utf-8")
print(f"✅ Generated {output_file} with {len(df)} rows.")
