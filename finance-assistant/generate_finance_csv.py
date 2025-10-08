import csv
import random
from datetime import datetime, timedelta

# File name
filename = "personal_finance_data.csv"

# Categories
expense_categories = ["Food", "Travel", "Bills", "Entertainment", "Shopping"]
income_categories = ["Salary", "Freelance", "Bonus"]

# Generate random transactions
def generate_transactions(num_days=30):
    data = []
    start_date = datetime.now() - timedelta(days=num_days)

    balance = 0
    for i in range(num_days):
        date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")

        # Randomly decide if it's income or expense
        if random.random() < 0.2:  # 20% chance for income
            category = random.choice(income_categories)
            amount = random.randint(10000, 30000) if category == "Salary" else random.randint(2000, 8000)
            txn_type = "Income"
            description = f"{category} credited"
            balance += amount
        else:
            category = random.choice(expense_categories)
            amount = random.randint(200, 3000)
            txn_type = "Expense"
            description = f"{category} expense"
            balance -= amount

        data.append([date, description, category, amount, txn_type, balance])

    return data


# Write to CSV
def write_csv(filename):
    headers = ["Date", "Description", "Category", "Amount", "Type", "Balance"]

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(generate_transactions())

    print(f"✅ CSV file '{filename}' generated successfully!")


if __name__ == "__main__":
    write_csv(filename)