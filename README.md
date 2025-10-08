# Personal Finance Assistant

The Personal Finance AI Assistant is an intelligent Streamlit web app that helps users manage, analyze, and visualize their financial activities — including expenses, savings, and investments.
It uses Machine Learning (ML) to auto-categorize bank transactions, generate budget insights, and analyze your investment portfolio.

The app provides:

🧾 Transaction categorization using AI (Tfidf + RandomForest)

💰 Personalized monthly budgeting & salary-aware recommendations

📊 Spending insights via visual charts

💼 Investment portfolio analysis with profit/loss tracking

🔥 Clean, intuitive dashboard UI built with Streamlit

# 🧩 Features

# 🪙 1. Transaction Categorization (AI-powered)

a) Upload raw bank statements (CSV)

b) Automatically detects columns like Date, Description, and Amount

c) Uses TfidfVectorizer + RandomForest to classify transactions into categories like Food, Bills, Shopping, Entertainment, etc.

d) Supports heuristic fallback if no labeled data is available

# 💰 2. Monthly Budget Planner

a) Enter your monthly salary

b) Upload your expense dataset

c) Generates an AI-driven spending plan that adjusts dynamically to your salary

d) Visualizes total spending, savings, and category distribution

e) Detects overspending and gives clear, actionable recommendations

# 💼 3. Investment Portfolio Analyzer

Upload your investment portfolio CSV with columns:
asset_name, category, amount_invested, current_value, date_invested

Calculates:

a) Total Invested, Current Value

b) Net Profit/Loss and Percentage

c) Best & Worst Performing Assets

d) Displays category-wise portfolio allocation (pie chart)

e) Shows profit/loss by asset with clean horizontal bar charts

# ⚙️ Installation

1️⃣ Clone the Repository

git clone https://github.com/yourusername/finance-ai.git
cd finance-ai

2️⃣ Create a Virtual Environment
python -m venv .venv
.\.venv\Scripts\activate   # (Windows)
# or
source .venv/bin/activate  # (Mac/Linux)

3️⃣ Install Requirements
pip install -r requirements.txt

4️⃣ Run the Streamlit App
streamlit run app.py

5️⃣ Open in Browser
Your app will launch automatically at:
👉 http://localhost:8501

# 🧮 How It Works

a) Data Loading – Reads flexible CSV formats, detects key columns automatically.

b) AI Categorization – Uses ML model (TfidfVectorizer + RandomForest) for text classification.

c) Dynamic Budgeting – Scales spending recommendations with salary input.

d) Visualization – Generates charts (pie, bar, horizontal bar) for intuitive insights.

_Developed by_

_Advaita S S_
_B.Sc Computer Science with Data Analytics_
