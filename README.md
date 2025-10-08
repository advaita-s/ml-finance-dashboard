# Personal Finance Assistant

The Personal Finance AI Assistant is an intelligent Streamlit web app that helps users manage, analyze, and visualize their financial activities — including expenses, savings, and investments.
It uses Machine Learning (ML) to auto-categorize bank transactions, generate budget insights, and analyze your investment portfolio.

# The app provides:

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
