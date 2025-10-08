"""
Personal Finance AI Assistant
================================

Handles:
 - Loading flexible CSV formats
 - Preprocessing + feature engineering
 - Auto-labeling (heuristics) if no category labels exist
 - Training a classifier
 - Prediction & exporting
 - Salary-aware budgeting helper for recommended spending plan
"""

# Standard imports
import re
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# persistence
import joblib


# -----------------------------
# Utilities & preprocessing
# -----------------------------

def load_transactions(csv_input):
    """
    Load transactions from CSV or DataFrame.
    Auto-detects columns and normalizes schema: [date, amount, description].
    """

    if isinstance(csv_input, pd.DataFrame):
        df = csv_input.copy()
    else:
        df = pd.read_csv(csv_input, dtype=str)

    df.columns = [c.strip().lower() for c in df.columns]

    # Candidate column names
    date_candidates = ['date', 'date/time', 'txn_date', 'transaction_date', 'timestamp', 'time']
    desc_candidates = ['description', 'desc', 'narration', 'details', 'memo', 'mode', 'merchant', 'sub category']
    amt_candidates = ['amount', 'amt', 'money', 'value', 'debit', 'credit', 'income/expense']

    # Auto-detect
    date_col = next((c for c in df.columns if c in date_candidates), None)
    desc_col = next((c for c in df.columns if c in desc_candidates), None)
    amt_col = next((c for c in df.columns if c in amt_candidates), None)

    if not date_col or not amt_col:
        raise ValueError(f"CSV format not recognized. Columns found: {list(df.columns)}")

    # If no description column, warn user → fallback to empty
    if not desc_col:
        print("[WARN] No description-like column found. Predictions may be poor.")
        df['description'] = ""
        desc_col = 'description'

    # Rename
    df = df.rename(columns={date_col: 'date', amt_col: 'amount', desc_col: 'description'})

    # Parse
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['amount'] = (
        df['amount']
        .astype(str)
        .str.replace(',', '', regex=True)
        .str.replace(r'[^\d\.\-]', '', regex=True)
    )
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

    # Drop invalids
    df = df.dropna(subset=['date', 'amount'])

    # Features
    df['amount_abs'] = df['amount'].abs()
    df['is_credit'] = df['amount'] > 0
    df['day_of_week'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['description_clean'] = df['description'].astype(str).str.lower().str.strip()

    return df


def extract_merchant(description: str) -> str:
    if not isinstance(description, str):
        return ''
    s = re.sub(r"\(.*?\)", '', description)
    tokens = re.findall(r"[a-zA-Z&]{2,}", s)
    return ' '.join(tokens[:4]).strip()


# -----------------------------
# Heuristic labeling
# -----------------------------

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    'food': ['starbucks', 'restaurant', 'cafe', 'mcdonald', 'burger', 'kfc', 'pizza'],
    'groceries': ['supermarket', 'grocery', 'walmart', 'aldi', 'lidl'],
    'travel': ['uber', 'lyft', 'airbnb', 'delta', 'indigo', 'train', 'hotel'],
    'bills': ['electric', 'gas', 'water', 'phone', 'internet', 'netflix', 'spotify'],
    'salary': ['payroll', 'salary', 'deposit', 'employer'],
    'transfer': ['transfer', 'upi', 'zelle'],
    'shopping': ['amazon', 'flipkart', 'macy', 'zara', 'store'],
    'entertainment': ['movie', 'cinema', 'concert', 'stadium']
}


def apply_heuristic_labels(df: pd.DataFrame) -> pd.Series:
    cats = []
    for _, r in df.iterrows():
        s = r['description_clean']
        m = extract_merchant(s)
        assigned = None
        for cat, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in s or kw in m:
                    assigned = cat
                    break
            if assigned:
                break
        if not assigned:
            assigned = 'other'
        cats.append(assigned)
    return pd.Series(cats, index=df.index)


# -----------------------------
# ML pipeline
# -----------------------------

def build_pipeline(extra_cat_cols: List[str], random_state: int = 42) -> Pipeline:
    text_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    num_transformer = Pipeline([('scaler', StandardScaler())])
    cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    transformers = [
        ('text', text_vectorizer, 'description_clean'),
        ('num', num_transformer, ['amount_abs']),
        ('dow', cat_transformer, ['day_of_week'])
    ]

    for col in extra_cat_cols:
        transformers.append((f'cat_{col}', cat_transformer, [col]))

    preprocessor = ColumnTransformer(transformers=transformers)

    clf = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    return Pipeline([('pre', preprocessor), ('clf', clf)])


# -----------------------------
# Training & prediction
# -----------------------------

def train_and_evaluate(df: pd.DataFrame, model_path: Optional[str] = None) -> Tuple[Pipeline, Dict]:
    df2 = df.copy()

    # Ensure labels
    if 'category' not in df2.columns:
        print('[INFO] No category column — using heuristics.')
        df2['category'] = apply_heuristic_labels(df2)
    else:
        df2['category'] = df2['category'].astype(str).str.lower().str.strip()

    # Ensure description_clean exists and is non-empty
    if 'description_clean' not in df2.columns and 'description' in df2.columns:
        df2['description_clean'] = df2['description'].astype(str).str.lower().str.strip()
    elif 'description_clean' not in df2.columns:
        df2['description_clean'] = ""

    # If column is completely empty → fill with dummy text
    if df2['description_clean'].replace("", pd.NA).isna().all():
        print("[WARN] No valid text data found, filling with placeholder.")
        df2['description_clean'] = "transaction"

    # Select features
    base_features = ['description_clean', 'amount_abs', 'day_of_week']
    extra_features = [c for c in ['mode', 'category', 'sub category', 'income/expense'] if c in df2.columns]
    feature_cols = list(dict.fromkeys(base_features + extra_features))

    X = df2[feature_cols]
    y = df2['category']

    # Safety check: dataset too small
    if len(df2) < 5:
        print(f"[WARN] Dataset too small ({len(df2)} rows). Training on full dataset without split.")
        pipeline = build_pipeline(extra_cat_cols=extra_features, random_state=42)
        pipeline.fit(X, y)
        if model_path:
            joblib.dump(pipeline, model_path)
            print(f'[INFO] Model saved to {model_path}')
        return pipeline, {}

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
    )

    pipeline = build_pipeline(extra_cat_cols=extra_features, random_state=42)

    print('[INFO] Training classifier...')
    pipeline.fit(X_train, y_train)

    # Evaluate
    if len(X_test) > 0:
        y_pred = pipeline.predict(X_test)
        print('\nClassification Report:\n', classification_report(y_test, y_pred, zero_division=0))
    else:
        print("[WARN] Test set empty. Skipping evaluation.")

    if model_path:
        joblib.dump(pipeline, model_path)
        print(f'[INFO] Model saved to {model_path}')

    return pipeline, {}


def predict_transactions(pipeline: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    # Match training features dynamically
    base_features = ['description_clean', 'amount_abs', 'day_of_week']
    extra_features = [c for c in ['mode', 'category', 'sub category', 'income/expense'] if c in df.columns]
    feature_cols = list(dict.fromkeys(base_features + extra_features))

    X = df[feature_cols]
    preds = pipeline.predict(X)
    out = df.copy()
    out['predicted_category'] = preds
    return out


# -----------------------------
# Budget & recommendations (salary-aware)
# -----------------------------

def _pick_category_column(df: pd.DataFrame) -> str:
    """
    Prefer predicted_category (post-ML). Fallback to category. Else create 'other'.
    """
    if 'predicted_category' in df.columns:
        return 'predicted_category'
    if 'category' in df.columns:
        return 'category'
    # if nothing exists, create a column so downstream code won't crash
    df['category'] = 'other'
    return 'category'


def compute_budget_plan(
    df: pd.DataFrame,
    monthly_income: float,
    savings_rate: float = 0.20,
    min_nonzero: float = 1e-9
) -> pd.DataFrame:
    """
    Compute a salary-aware recommended spending plan that reacts to the given salary.

    Robustly detects expenses when:
      - amounts are negative for debits (standard)
      - OR all amounts are positive (some exports)
      - OR an 'income/expense' column exists
      - OR only categories indicate income
    """
    if monthly_income is None or not np.isfinite(monthly_income) or monthly_income < 0:
        raise ValueError("monthly_income must be a non-negative finite number.")

    cat_col = _pick_category_column(df)
    data = df.copy()

    # Helper
    def _norm_str(s):
        return str(s).strip().lower() if pd.notna(s) else ""

    # Detect expense rows
    has_neg = (data['amount'] < 0).any()

    if has_neg:
        expense_mask = data['amount'] < 0
    else:
        if 'income/expense' in data.columns:
            col = data['income/expense'].map(_norm_str)
            expense_mask = col.str.contains('exp')
        else:
            income_like = {'salary', 'income', 'deposit', 'payroll', 'employer', 'transfer'}
            if cat_col in data.columns:
                expense_mask = ~data[cat_col].map(_norm_str).isin(income_like)
            else:
                expense_mask = pd.Series(True, index=data.index)

    # Fallback if everything evaluated as income
    if not expense_mask.any():
        income_words = ('salary', 'income', 'deposit', 'payroll', 'employer', 'transfer')
        expense_mask = ~data[cat_col].map(_norm_str).str.contains('|'.join(income_words))

    # Build expense table
    exp = data.loc[expense_mask].copy()
    if exp.empty:
        target_savings = round(float(monthly_income * savings_rate), 2)
        spendable_budget = round(float(monthly_income - target_savings), 2)
        return pd.DataFrame({
            'category': [],
            'current_spend': [],
            'recommended_spend': [],
            'delta': []
        }).assign(target_savings=target_savings, spendable_budget=spendable_budget)

    exp['expense_abs'] = exp['amount'].abs()
    current_by_cat = exp.groupby(cat_col, dropna=False)['expense_abs'].sum().sort_values(ascending=False)

    total_current_exp = float(current_by_cat.sum())
    target_savings = float(monthly_income * savings_rate)
    spendable_budget = float(max(monthly_income - target_savings, 0.0))

    if total_current_exp <= min_nonzero:
        recommended = current_by_cat * 0.0
    else:
        shares = current_by_cat / (total_current_exp + min_nonzero)
        recommended = shares * spendable_budget

    # Normalize to match spendable_budget exactly (avoid drift)
    if recommended.sum() > 0:
        recommended *= (spendable_budget / float(recommended.sum()))

    result = pd.DataFrame({
        'category': current_by_cat.index.astype(str),
        'current_spend': current_by_cat.values.astype(float),
        'recommended_spend': recommended.values.astype(float),
    })
    result['current_spend'] = result['current_spend'].round(2)
    result['recommended_spend'] = result['recommended_spend'].round(2)
    result['delta'] = (result['recommended_spend'] - result['current_spend']).round(2)
    result['target_savings'] = round(target_savings, 2)
    result['spendable_budget'] = round(spendable_budget, 2)

    return result.sort_values('recommended_spend', ascending=False).reset_index(drop=True)


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str)
    parser.add_argument('--predict', type=str)
    parser.add_argument('--model-out', type=str, default='pfa_model.joblib')
    args = parser.parse_args()

    model = None
    if args.train:
        df_train = load_transactions(args.train)
        model, _ = train_and_evaluate(df_train, model_path=args.model_out)

    if args.predict:
        df_pred = load_transactions(args.predict)
        if model is None:
            model = joblib.load(args.model_out)
        out = predict_transactions(model, df_pred)
        out_path = Path(args.predict).stem + '_predicted.csv'
        out.to_csv(out_path, index=False)
        print(f"[INFO] Predictions saved to {out_path}")


if __name__ == '__main__':
    main()
