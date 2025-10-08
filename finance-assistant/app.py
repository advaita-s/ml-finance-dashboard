import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from personal_finance_ai_assistant import compute_budget_plan


from personal_finance_ai_assistant import (
    load_transactions,
    predict_transactions,
    extract_merchant,
    train_and_evaluate
)

# ==============================
# APP HEADER
# ==============================
st.set_page_config(layout="wide")
st.title("💰 Personal Finance Assistant")
st.write("Choose between analyzing raw bank transactions, planning your monthly savings, or analyzing investments.")

# ==============================
# MAIN TABS
# ==============================
tab1, tab2, tab3 = st.tabs([
    "🏦 Transaction Categorizer",
    "📅 Monthly Budget Planner",
    "📊 Investment Analyzer"
])

# -------------------------------
# TAB 1: Transaction Categorizer
# -------------------------------
with tab1:
    st.header("🏦 Upload Bank Transactions")
    st.markdown("""
    **📌 Dataset Requirements:**  
    - Must have these columns (auto-detected or selectable):  
      - `date` (or similar like `txn_date`, `timestamp`)  
      - `amount` (or similar like `debit`, `credit`, `money`)  
      - `description` (or similar like `narration`, `details`, `memo`)  
    """)

    uploaded_file = st.file_uploader("Upload Bank Transactions CSV", type=["csv"], key="bank_csv")
    model_file = st.file_uploader("Optional: upload a pre-trained model (.joblib)", type=["joblib", "pkl"], key="bank_model")

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file, dtype=str)
        raw_df.columns = [c.strip().lower() for c in raw_df.columns]

        # Column mapping
        date_candidates = ["date", "date/time", "txn_date", "transaction_date", "timestamp", "time"]
        amt_candidates = ["amount", "amt", "money", "value", "debit", "credit", "income/expense"]
        desc_candidates = ["description", "desc", "narration", "details", "memo", "mode", "merchant", "sub category"]

        date_col = next((c for c in raw_df.columns if c in date_candidates), None)
        amt_col = next((c for c in raw_df.columns if c in amt_candidates), None)
        desc_col = next((c for c in raw_df.columns if c in desc_candidates), None)

        if not date_col:
            date_col = st.selectbox("Select Date Column", options=raw_df.columns)
        if not amt_col:
            amt_col = st.selectbox("Select Amount Column", options=raw_df.columns)
        if not desc_col:
            st.warning("⚠️ No description-like column found. Please select one manually.")
            desc_col = st.selectbox("Select Column for Transaction Details", options=raw_df.columns)

        multi_desc_cols = st.multiselect(
            "Optional: Merge additional columns into description",
            options=[c for c in raw_df.columns if c != desc_col]
        )

        df = raw_df.copy()
        df = df.rename(columns={date_col: "date", amt_col: "amount", desc_col: "description"})
        if multi_desc_cols:
            df["description"] = df["description"].astype(str) + " " + df[multi_desc_cols].astype(str).agg(" ".join, axis=1)

        try:
            df = load_transactions(df)
        except Exception as e:
            st.error(f"⚠️ Could not process dataset: {e}")
            st.stop()

        st.subheader("Preview of Uploaded Transactions")
        st.dataframe(df.head(50))

        # Model
        if model_file is not None:
            pipeline = joblib.load(model_file)
            st.success("✅ Loaded model from file")
        else:
            st.info("⚙️ Training a fresh model using heuristic labels.")
            pipeline, _ = train_and_evaluate(df, model_path=None)

        # Predictions
        out = predict_transactions(pipeline, df)
        st.success("🎉 Predictions complete — preview below")
        st.dataframe(out[["date", "description", "amount", "predicted_category"]].head(200))

        # Spending by category
        st.subheader("📊 Spending by Predicted Category")
        fig1 = plt.figure(figsize=(6, 3))
        s = out.groupby("predicted_category")["amount_abs"].sum().sort_values(ascending=False)
        s.plot(kind="bar")
        plt.ylabel("Total spent (absolute)")
        plt.tight_layout()
        st.pyplot(fig1)

        # Monthly trend
        st.subheader("📈 Monthly Spending Trend")
        out["month"] = out["date"].dt.to_period("M")
        fig2 = plt.figure(figsize=(8, 3))
        ts = out.groupby("month")["amount_abs"].sum().sort_index()
        ts.plot(marker="o")
        plt.ylabel("Amount (absolute)")
        plt.xlabel("Month")
        plt.tight_layout()
        st.pyplot(fig2)

        # Merchants
        st.subheader("🏪 Top Merchants")
        out["merchant"] = out["description"].apply(extract_merchant)
        top_merchants = out.groupby("merchant")["amount_abs"].sum().sort_values(ascending=False).head(15)
        st.table(top_merchants)

        # Summary Insights
        st.subheader("✨ Summary Insights")
        try:
            total_spent = out.loc[out["amount"] < 0, "amount_abs"].sum()
            total_income = out.loc[out["amount"] > 0, "amount_abs"].sum()
            top_cat = out.groupby("predicted_category")["amount_abs"].sum().sort_values(ascending=False).head(3)
            top_merchant = out.groupby("merchant")["amount_abs"].sum().sort_values(ascending=False).head(1)
            biggest_txn = out.loc[out["amount_abs"].idxmax()]

            st.markdown(f"""
            - 💸 **Total Spending:** {total_spent:,.2f}  
            - 💰 **Total Income/Credits:** {total_income:,.2f}  
            - 📊 **Top Categories:** {', '.join(top_cat.index.tolist())}  
            - 🏪 **Top Merchant:** {top_merchant.index[0]} (₹{top_merchant.iloc[0]:,.2f})  
            - ⚡ **Biggest Transaction:** {biggest_txn['description']} (₹{biggest_txn['amount_abs']:,.2f})
            """)
        except Exception as e:
            st.warning(f"Could not generate insights: {e}")

        # Download
        st.download_button(
            "⬇️ Download Predictions CSV",
            out.to_csv(index=False).encode("utf-8"),
            file_name="predicted_transactions.csv"
        )

# -------------------------------
# TAB 2: Monthly Budget Planner
# -------------------------------
# -------------------------------
# TAB 2: Monthly Budget Planner
# -------------------------------
with tab2:
    st.header("📅 Monthly Budget Planner")
    st.write("Upload your monthly spending dataset (CSV) and enter your salary to get personalized insights.")

    salary_input = st.number_input("Enter your Monthly Salary (₹)", min_value=1000.0, step=1000.0)
    exp_file = st.file_uploader("Upload your Expense CSV", type=["csv"], key="planner")

    if exp_file is not None and salary_input > 0:
        df_exp = pd.read_csv(exp_file)
        df_exp.columns = [c.strip().lower() for c in df_exp.columns]

        if "amount" not in df_exp.columns or "category" not in df_exp.columns:
            st.error("Dataset must contain at least `amount` and `category` columns.")
            st.stop()

        # Clean amounts
        df_exp["amount"] = (
            df_exp["amount"]
            .astype(str)
            .str.replace(",", "", regex=True)
            .str.replace(r"[^\d\.\-]", "", regex=True)
        )
        df_exp["amount"] = pd.to_numeric(df_exp["amount"], errors="coerce").fillna(0)

        # Separate income vs expense
        expense_df = df_exp[~df_exp["category"].str.contains("salary|income|deposit|credit|bonus",
                                                             case=False, na=False)]
        total_spent = expense_df["amount"].sum()
        total_income = salary_input
        total_savings = total_income - total_spent

        highest_expense_cat = expense_df.groupby("category")["amount"].sum().idxmax() if not expense_df.empty else "N/A"
        lowest_expense_cat = expense_df.groupby("category")["amount"].sum().idxmin() if not expense_df.empty else "N/A"

        st.subheader("✨ Summary Insights")
        st.markdown(f"""
        - 💸 **Total Spending:** ₹{total_spent:,.2f}  
        - 💰 **Total Savings:** ₹{total_savings:,.2f}  
        - 📈 **Highest Expense Category:** {highest_expense_cat}  
        - 📉 **Lowest Expense Category:** {lowest_expense_cat}  
        """)

        if total_spent > total_income:
            st.error(f"🚨 Overspending Risk: Your spending ₹{total_spent:,.2f} exceeds your salary ₹{total_income:,.2f}")

        if not expense_df.empty:
            st.subheader("🥧 Spending Distribution by Category")
            fig_pie, ax = plt.subplots()
            expense_df.groupby("category")["amount"].sum().plot.pie(
                ax=ax, autopct="%1.1f%%", figsize=(5, 5), startangle=90
            )
            ax.set_ylabel("")
            st.pyplot(fig_pie)

        # 🧾 Recommended Spending Plan (PASTE HERE)
        st.subheader("📝 Recommended Spending Plan")

        plan = compute_budget_plan(
            df_exp,                      # defined inside this if-block
            monthly_income=float(salary_input),
            savings_rate=0.20
        )

        if len(plan) == 0:
            st.info("No expenses detected to allocate. Add transactions first.")
        else:
            st.caption(
                f"Target Savings: ₹{plan['target_savings'].iloc[0]:,.2f} | "
                f"Spendable Budget: ₹{plan['spendable_budget'].iloc[0]:,.2f}"
            )

            for _, r in plan.iterrows():
                st.markdown(
                    f"⚡ Reduce **{r['category'].title()}** spending to ~₹{r['recommended_spend']:,.2f} "
                    f"to balance your budget. (Δ ₹{r['delta']:,.2f})"
                )

            st.code(
                f"Check: sum(recommended) = ₹{plan['recommended_spend'].sum():,.2f} "
                f"≈ Spendable Budget ₹{plan['spendable_budget'].iloc[0]:,.2f}",
                language="text"
            )


# -------------------------------
# TAB 3: Investment Portfolio Analyzer
# -------------------------------
with tab3:
    st.header("📊 Investment Portfolio Analyzer")
    st.write("Upload your investment portfolio CSV to analyze performance and diversification.")
    st.markdown("_Required columns: `asset_name`, `category`, `amount_invested`, `current_value`, `date_invested`_")

    inv_file = st.file_uploader("Upload Investment CSV", type=["csv"], key="inv")

    if inv_file is not None:
        df_inv = pd.read_csv(inv_file)
        required_cols = ["asset_name", "category", "amount_invested", "current_value", "date_invested"]
        missing_cols = [c for c in required_cols if c not in df_inv.columns]

        if missing_cols:
            st.error(f"⚠️ Missing required columns: {missing_cols}")
        else:
            # Calculate profit/loss
            df_inv["profit_loss"] = df_inv["current_value"] - df_inv["amount_invested"]
            df_inv["profit_loss_pct"] = (df_inv["profit_loss"] / df_inv["amount_invested"]) * 100

            # Overall summary
            total_invested = df_inv["amount_invested"].sum()
            total_value = df_inv["current_value"].sum()
            net_profit = total_value - total_invested
            net_profit_pct = (net_profit / total_invested) * 100 if total_invested > 0 else 0

            best_asset = df_inv.loc[df_inv["profit_loss_pct"].idxmax()]
            worst_asset = df_inv.loc[df_inv["profit_loss_pct"].idxmin()]

            st.subheader("✨ Portfolio Insights")
            st.markdown(f"""
            - 💵 **Total Invested:** ₹{total_invested:,.2f}  
            - 📈 **Current Value:** ₹{total_value:,.2f}  
            - 🟢 **Net Profit/Loss:** ₹{net_profit:,.2f} ({net_profit_pct:.2f}%)  
            - 🥇 **Best Performing Asset:** {best_asset['asset_name']} ({best_asset['profit_loss_pct']:.2f}%)  
            - ⚠️ **Worst Performing Asset:** {worst_asset['asset_name']} ({worst_asset['profit_loss_pct']:.2f}%)  
            """)

            # -------------------------------
            # PIE CHART - Portfolio Allocation
            # -------------------------------
            st.subheader("📊 Portfolio Allocation by Category")
            fig1, ax1 = plt.subplots(figsize=(6, 3))
            df_inv.groupby("category")["current_value"].sum().plot(
                kind="pie", autopct="%1.1f%%", ax=ax1, startangle=90
            )
            ax1.set_ylabel("")
            ax1.set_title("Category-wise Allocation", fontsize=11, pad=10)
            plt.tight_layout()
            st.pyplot(fig1)

            # -------------------------------
            # FIXED SECTION - Profit/Loss by Asset (Main Issue)
            # -------------------------------
            st.subheader("📉 Profit/Loss by Asset")

            # Sort assets for better readability
            df_sorted = df_inv.sort_values("profit_loss", ascending=True)

            # ✅ Horizontal bar chart - perfect for long labels
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.barh(df_sorted["asset_name"], df_sorted["profit_loss"], color="#008b8b")

            # Add titles & labels
            ax2.set_title("Profit/Loss by Asset", fontsize=12, pad=15)
            ax2.set_xlabel("Profit / Loss (₹)")
            ax2.set_ylabel("Asset Name")

            # Highlight the zero line
            ax2.axvline(0, color="red", linestyle="--", linewidth=1)

            # Tight layout to prevent label clipping
            plt.tight_layout()
            st.pyplot(fig2)

            # Optional: Display summary table below
            st.dataframe(df_inv[["asset_name", "category", "amount_invested", "current_value", "profit_loss", "profit_loss_pct"]])
