import csv
import random
from datetime import datetime, timedelta

# File name
filename = "investment_portfolio.csv"

# Investment options
assets = [
    ("Tata Steel", "Equity", "High"),
    ("Infosys", "Equity", "Medium"),
    ("HDFC Bank", "Equity", "Medium"),
    ("Reliance Jio Bond", "Debt", "Low"),
    ("Government Bond", "Debt", "Low"),
    ("Bitcoin", "Crypto", "High"),
    ("Ethereum", "Crypto", "High"),
    ("Gold ETF", "Commodity", "Medium"),
    ("Real Estate Fund", "Real Estate", "Low"),
    ("Mutual Fund Growth", "Mutual Fund", "Medium")
]

# Generate random investment records
def generate_investments(num_records=30):
    data = []
    start_date = datetime.now() - timedelta(days=num_records)

    for i in range(num_records):
        date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
        asset, inv_type, risk = random.choice(assets)

        amount_invested = random.randint(5000, 50000)
        # Simulate current value with small random growth/decline
        current_value = round(amount_invested * random.uniform(0.9, 1.3), 2)
        profit_loss = round(current_value - amount_invested, 2)
        return_pct = round((profit_loss / amount_invested) * 100, 2)

        data.append([date, asset, inv_type, amount_invested, current_value, profit_loss, return_pct, risk])

    return data

# Write to CSV
def write_csv(filename):
    headers = ["Date", "Asset", "Investment Type", "Amount Invested", "Current Value", "Profit/Loss", "Return %", "Risk Level"]

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(generate_investments())

    print(f"✅ CSV file '{filename}' generated successfully!")

if __name__ == "__main__":
    write_csv(filename)
