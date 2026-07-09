"""
Lost Client Prediction — end-to-end pipeline + Streamlit dashboard
Run:      streamlit run churn_app.py
Install:  pip install streamlit pandas numpy scikit-learn imbalanced-learn xgboost plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                              recall_score, f1_score, fbeta_score, roc_auc_score,
                              average_precision_score, matthews_corrcoef, confusion_matrix,
                              roc_curve, precision_recall_curve, make_scorer)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMB_AVAILABLE = True
except ImportError:
    IMB_AVAILABLE = False

st.set_page_config(page_title="Lost Client Prediction", layout="wide")

# =========================================================
# CONFIG - adjust to your real column names if different
# =========================================================
CFG = dict(
    date_col="Date", type_col="Type", sale_val=10, return_val=12,
    client_id="Oid-2", client_code="Code-2", client_name="Label1-2",
    client_cat="Label1-3", client_region="Label1-4", client_wilaya="Label1-5",
    prod_code="Code", qty="Quantity", amount="Amount", vwap="VWAP", disc="DiscountPercent",
)

CAT_FEATS = ["client_cat", "client_region", "client_wilaya"]
NUM_FEATS = [
    "recency_days", "frequency", "tenure_days", "monetary_total", "monetary_avg",
    "avg_vwap", "avg_discount", "total_qty", "n_products", "return_rate",
    "amt_last_30", "amt_last_60", "amt_last_90", "cnt_last_30", "cnt_last_60", "cnt_last_90",
    "avg_interpurchase_days", "trend_ratio",
]

# =========================================================
# DATA LOADING & FEATURE ENGINEERING
# =========================================================
@st.cache_data(show_spinner=False)
def load_data(file):
    df = pd.read_csv(file)
    df[CFG["date_col"]] = pd.to_datetime(df[CFG["date_col"]], errors="coerce", dayfirst=True)
    df = df.dropna(subset=[CFG["date_col"], CFG["client_id"]])
    return df

def _safe_ratio(a, b):
    return a / b if b not in (0, None) and not pd.isna(b) else 0.0

def compute_client_features(df, cutoff_date, config=CFG):
    """One row per client, built ONLY from data with Date <= cutoff_date."""
    d = df[df[config["date_col"]] <= cutoff_date].copy()
    if d.empty:
        return pd.DataFrame()

    d["signed_amount"] = np.where(d[config["type_col"]] == config["return_val"],
                                   -d[config["amount"]], d[config["amount"]])

    rows = []
    for cid, g in d.groupby(config["client_id"]):
        g_sales = g[g[config["type_col"]] == config["sale_val"]]
        g_ret = g[g[config["type_col"]] == config["return_val"]]

        if g_sales.empty:
            last_purchase = g[config["date_col"]].max()
            first_purchase = g[config["date_col"]].min()
        else:
            last_purchase = g_sales[config["date_col"]].max()
            first_purchase = g_sales[config["date_col"]].min()

        recency_days = (cutoff_date - last_purchase).days
        tenure_days = max((cutoff_date - first_purchase).days, 0)
        frequency = g_sales[config["date_col"]].nunique()
        monetary_total = g["signed_amount"].sum()
        monetary_avg = g_sales[config["amount"]].mean() if len(g_sales) else 0.0
        avg_vwap = g_sales[config["vwap"]].mean() if len(g_sales) else 0.0
        avg_discount = g_sales[config["disc"]].mean() if len(g_sales) else 0.0
        total_qty = g_sales[config["qty"]].sum() if len(g_sales) else 0.0
        n_products = g_sales[config["prod_code"]].nunique() if len(g_sales) else 0
        n_returns, n_sales = len(g_ret), len(g_sales)
        return_rate = _safe_ratio(n_returns, n_sales + n_returns)

        last30 = g[g[config["date_col"]] > cutoff_date - timedelta(days=30)]
        last60 = g[g[config["date_col"]] > cutoff_date - timedelta(days=60)]
        last90 = g[g[config["date_col"]] > cutoff_date - timedelta(days=90)]

        purchase_dates = np.sort(g_sales[config["date_col"]].unique())
        if len(purchase_dates) > 1:
            gaps = np.diff(purchase_dates).astype("timedelta64[D]").astype(int)
            avg_gap = gaps.mean()
        else:
            avg_gap = tenure_days

        mid = first_purchase + (cutoff_date - first_purchase) / 2
        first_half = g_sales[g_sales[config["date_col"]] <= mid][config["amount"]].sum()
        second_half = g_sales[g_sales[config["date_col"]] > mid][config["amount"]].sum()
        trend_ratio = (second_half / first_half if first_half > 0
                        else (1.0 if second_half == 0 else 2.0))

        rows.append({
            "client_id": cid,
            "client_code": g[config["client_code"]].iloc[0],
            "client_name": g[config["client_name"]].iloc[0],
            "client_cat": g[config["client_cat"]].iloc[0],
            "client_region": g[config["client_region"]].iloc[0],
            "client_wilaya": g[config["client_wilaya"]].iloc[0],
            "recency_days": recency_days,
            "frequency": frequency,
            "tenure_days": tenure_days,
            "monetary_total": monetary_total,
            "monetary_avg": monetary_avg,
            "avg_vwap": avg_vwap,
            "avg_discount": avg_discount,
            "total_qty": total_qty,
            "n_products": n_products,
            "return_rate": return_rate,
            "amt_last_30": last30[config["amount"]].sum(),
            "amt_last_60": last60[config["amount"]].sum(),
            "amt_last_90": last90[config["amount"]].sum(),
            "cnt_last_30": last30.shape[0],
            "cnt_last_60": last60.shape[0],
            "cnt_last_90": last90.shape[0],
            "avg_interpurchase_days": avg_gap,
            "trend_ratio": trend_ratio,
        })

    return pd.DataFrame(rows)

def add_label(feat_df, threshold_days=100):
    feat_df = feat_df.copy()
    feat_df["lost"] = (feat_df["recency_days"] >= threshold_days).astype(int)
    return feat_df

# =========================================================
# MODELING
# =========================================================
def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS),
    ])

def get_model_grid():
    models = {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l2"]}
        ),
        "RandomForest": (
            RandomForestClassifier(class_weight="balanced", random_state=42),
            {"clf__n_estimators": [200, 400, 600], "clf__max_depth": [4, 8, 12, None],
             "clf__min_samples_leaf": [1, 3, 5]}
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=42),
            {"clf__n_estimators": [100, 200, 300], "clf__learning_rate": [0.01, 0.05, 0.1],
             "clf__max_depth": [2, 3, 4]}
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [5, 11, 21, 31], "clf__weights": ["uniform", "distance"]}
        ),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = (
            XGBClassifier(eval_metric="logloss", random_state=42),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 5, 7],
             "clf__learning_rate": [0.01, 0.05, 0.1], "clf__scale_pos_weight": [1, 3, 5]}
        )
    return models

def _grid_size(grid):
    size = 1
    for v in grid.values():
        size *= len(v)
    return size

def train_and_tune(X_train, y_train, scoring_beta=0.5, n_iter=12, use_smote=False):
    """
    scoring_beta < 1 weights PRECISION more than recall during tuning.
    Business rule: flagging an active client as 'lost' (False Positive) is costlier
    than missing a truly lost client (False Negative) -> favor precision.
    """
    fbeta_scorer = make_scorer(fbeta_score, beta=scoring_beta, zero_division=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, (estimator, grid) in get_model_grid().items():
        preproc = build_preprocessor()
        if use_smote and IMB_AVAILABLE:
            pipe = ImbPipeline([("prep", preproc), ("smote", SMOTE(random_state=42)), ("clf", estimator)])
        else:
            pipe = Pipeline([("prep", preproc), ("clf", estimator)])
        search = RandomizedSearchCV(pipe, grid, n_iter=min(n_iter, _grid_size(grid)),
                                     scoring=fbeta_scorer, cv=cv, random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        results[name] = search.best_estimator_
    return results

def evaluate_model(model, X_test, y_test, threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, pred),
        "Precision (lost)": precision_score(y_test, pred, zero_division=0),
        "Recall (lost)": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "F0.5 (precision-weighted)": fbeta_score(y_test, pred, beta=0.5, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "PR-AUC (avg precision)": average_precision_score(y_test, proba),
        "MCC": matthews_corrcoef(y_test, pred),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "_proba": proba, "_pred": pred,
    }

def get_feature_names(preprocessor):
    cat = list(preprocessor.named_transformers_["cat"].get_feature_names_out(CAT_FEATS))
    return NUM_FEATS + cat

def get_feature_importance(model, feature_names):
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_[0])
    else:
        return pd.DataFrame()
    return pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values(
        "importance", ascending=False)

# =========================================================
# STREAMLIT APP
# =========================================================
def main():
    st.title("🔻 Lost Client Prediction Dashboard")

    st.sidebar.header("1. Data")
    file = st.sidebar.file_uploader("Upload transactions CSV", type=["csv"])
    if file is None:
        st.info("Upload your transactions CSV to start (columns: Oid, Code, Label1, Quantity, "
                 "Amount, VWAP, DiscountPercent, Date, Type, Oid-2, Code-2, Label1-2, Label1-3, "
                 "Label1-4, Label1-5).")
        st.stop()

    df = load_data(file)

    st.sidebar.header("2. Labeling rules")
    cutoff_date = pd.Timestamp(st.sidebar.date_input("Cutoff date (train boundary)",
                                                       value=pd.Timestamp("2026-01-01")))
    lost_days = st.sidebar.number_input("Days since last purchase = lost", value=100, min_value=1)

    st.sidebar.header("3. Training")
    test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2)
    use_smote = st.sidebar.checkbox("Use SMOTE oversampling", value=False)
    beta = st.sidebar.slider("Precision weight (Fbeta) - lower = more precision-focused", 0.3, 1.5, 0.5)
    threshold = st.sidebar.slider("Decision threshold (probability of 'lost')", 0.05, 0.95, 0.5)

    tabs = st.tabs(["📊 Overview", "🏷️ Labels", "🤖 Models", "⭐ Importance", "🔮 Predict New Data"])

    # ---- Overview ----
    with tabs[0]:
        st.subheader("Raw data")
        st.dataframe(df.head(20))
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", len(df))
        c2.metric("Clients", df[CFG["client_id"]].nunique())
        c3.metric("Date range", f"{df[CFG['date_col']].min().date()} → {df[CFG['date_col']].max().date()}")
        st.plotly_chart(px.histogram(df, x=CFG["date_col"], title="Transactions over time"),
                         use_container_width=True)

    # ---- Labels / features on part 1 ----
    train_full = df[df[CFG["date_col"]] <= cutoff_date]
    feats = compute_client_features(train_full, cutoff_date)
    feats = add_label(feats, lost_days)

    with tabs[1]:
        st.subheader(f"Client features as of {cutoff_date.date()} (Part 1: data ≤ cutoff)")
        st.dataframe(feats.head(20))
        dist = feats["lost"].value_counts().rename({0: "Not lost", 1: "Lost"})
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.pie(values=dist.values, names=dist.index, title="Class balance"),
                         use_container_width=True)
        c2.metric("Lost rate", f"{feats['lost'].mean()*100:.1f}%")

    # ---- Train models ----
    with tabs[2]:
        st.subheader("Train & compare classification models")
        st.caption("Predicting a client as 'lost' when he is actually active (False Positive) is "
                   "treated as MORE costly than missing a truly lost client (False Negative), so "
                   "tuning favors precision via an F-beta score (beta < 1).")
        if st.button("🚀 Train models"):
            X = feats[NUM_FEATS + CAT_FEATS]
            y = feats["lost"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42)

            with st.spinner("Tuning models..."):
                models = train_and_tune(X_train, y_train, scoring_beta=beta, use_smote=use_smote)

            results = {name: evaluate_model(m, X_test, y_test, threshold) for name, m in models.items()}
            st.session_state["models"] = models
            st.session_state["results"] = results
            st.session_state["y_test"] = y_test

        if "results" in st.session_state:
            results = st.session_state["results"]
            comp = pd.DataFrame({name: {k: v for k, v in r.items() if not k.startswith("_")}
                                  for name, r in results.items()}).T
            sort_metric = st.selectbox("Sort by", comp.columns.tolist(),
                                        index=comp.columns.get_loc("F0.5 (precision-weighted)"))
            comp = comp.sort_values(sort_metric, ascending=False)
            st.dataframe(comp.style.background_gradient(cmap="Greens", subset=[sort_metric]))

            best_name = comp.index[0]
            st.success(f"Best model: **{best_name}**")
            st.session_state["best_model_name"] = best_name

            metric_cols = ["Precision (lost)", "Recall (lost)", "F1", "F0.5 (precision-weighted)",
                            "ROC-AUC", "PR-AUC (avg precision)"]
            fig = go.Figure()
            for name in comp.index:
                fig.add_trace(go.Bar(name=name, x=metric_cols, y=[results[name][m] for m in metric_cols]))
            fig.update_layout(barmode="group", title="Metric comparison")
            st.plotly_chart(fig, use_container_width=True)

            sel = st.selectbox("Inspect model", comp.index.tolist())
            r = results[sel]
            c1, c2 = st.columns(2)
            with c1:
                cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
                st.plotly_chart(px.imshow(cm, text_auto=True, x=["Pred Not lost", "Pred Lost"],
                                           y=["Actual Not lost", "Actual Lost"],
                                           title=f"Confusion matrix — {sel}"), use_container_width=True)
            with c2:
                fpr, tpr, _ = roc_curve(st.session_state["y_test"], r["_proba"])
                prec, rec, _ = precision_recall_curve(st.session_state["y_test"], r["_proba"])
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=fpr, y=tpr, name="ROC"))
                fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash"), name="Random"))
                fig2.update_layout(title=f"ROC curve — {sel}", xaxis_title="FPR", yaxis_title="TPR")
                st.plotly_chart(fig2, use_container_width=True)
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=rec, y=prec, name="PR curve"))
                fig3.update_layout(title=f"Precision-Recall curve — {sel}",
                                    xaxis_title="Recall", yaxis_title="Precision")
                st.plotly_chart(fig3, use_container_width=True)

    # ---- Feature importance ----
    with tabs[3]:
        if "models" not in st.session_state:
            st.warning("Train models first (tab 'Models').")
        else:
            sel = st.selectbox("Model", list(st.session_state["models"].keys()), key="fi_sel")
            model = st.session_state["models"][sel]
            fnames = get_feature_names(model.named_steps["prep"])
            fi = get_feature_importance(model, fnames)
            if fi.empty:
                st.info(f"{sel} does not expose feature importances/coefficients directly.")
            else:
                st.plotly_chart(px.bar(fi.head(20), x="importance", y="feature", orientation="h",
                                        title=f"Top 20 important features — {sel}"
                                        ).update_yaxes(categoryorder="total ascending"),
                                 use_container_width=True)

    # ---- Predict on the remaining data ----
    with tabs[4]:
        st.subheader("Predict lost clients on the remaining data (after cutoff)")
        if "models" not in st.session_state or "best_model_name" not in st.session_state:
            st.warning("Train models first (tab 'Models').")
        else:
            as_of = df[CFG["date_col"]].max()
            st.write(f"Computing features for all clients using full history up to "
                     f"**{as_of.date()}** (dataset's latest date).")
            model_names = list(st.session_state["models"].keys())
            model_name = st.selectbox("Model to use for prediction", model_names,
                                       index=model_names.index(st.session_state["best_model_name"]))
            model = st.session_state["models"][model_name]

            new_feats = compute_client_features(df, as_of)
            X_new = new_feats[NUM_FEATS + CAT_FEATS]
            proba = model.predict_proba(X_new)[:, 1]
            new_feats["lost_probability"] = proba
            new_feats["predicted_status"] = np.where(proba >= threshold, "Lost", "Not lost")

            c1, c2, c3 = st.columns(3)
            c1.metric("Clients", len(new_feats))
            c2.metric("Predicted lost", int((new_feats["predicted_status"] == "Lost").sum()))
            c3.metric("Predicted lost rate",
                      f"{(new_feats['predicted_status']=='Lost').mean()*100:.1f}%")

            f1, f2, f3 = st.columns(3)
            region_f = f1.multiselect("Region", sorted(new_feats["client_region"].dropna().unique()))
            wilaya_f = f2.multiselect("Wilaya", sorted(new_feats["client_wilaya"].dropna().unique()))
            cat_f = f3.multiselect("Category", sorted(new_feats["client_cat"].dropna().unique()))

            view = new_feats.copy()
            if region_f: view = view[view["client_region"].isin(region_f)]
            if wilaya_f: view = view[view["client_wilaya"].isin(wilaya_f)]
            if cat_f: view = view[view["client_cat"].isin(cat_f)]

            st.dataframe(view.sort_values("lost_probability", ascending=False)[
                ["client_id", "client_code", "client_name", "client_cat", "client_region",
                 "client_wilaya", "recency_days", "frequency", "monetary_total",
                 "lost_probability", "predicted_status"]])

            csv = view.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download predictions CSV", csv, "predicted_lost_clients.csv", "text/csv")

if __name__ == "__main__":
    main()
    
