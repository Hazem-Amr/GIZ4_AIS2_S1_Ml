"""
Boston Housing Price Predictor — Streamlit App
================================================
Reproduces the full ML pipeline from *boston_housing_model.ipynb* and exposes
it through an interactive web UI.

Models available:
    • Linear Regression
    • Decision Tree  (GridSearchCV, max_depth tuned)
    • Random Forest  (n_estimators=100, max_depth=5)
    • Gradient Boosting (n_estimators=100, max_depth=5)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor

# ──────────────────────────── constants ────────────────────────────
FEATURES = ["RM", "LSTAT", "PTRATIO"]
TARGET = "MEDV"

FEATURE_DESCRIPTIONS = {
    "RM": "Average number of rooms per dwelling",
    "LSTAT": "% of lower-status workers in the neighbourhood",
    "PTRATIO": "Student-to-teacher ratio",
}

MODEL_NAMES = [
    "Linear Regression",
    "Decision Tree (GridSearchCV)",
    "Random Forest",
    "Gradient Boosting",
]


# ──────────────────────────── data / model helpers ─────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the Boston housing CSV that lives alongside this script."""
    csv_path = Path(__file__).resolve().parent / "housing.csv"
    return pd.read_csv(csv_path)


@st.cache_resource
def train_all_models():
    """Train every model used in the notebook and return them with metrics."""
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # --- Decision Tree with GridSearchCV (mirrors notebook) ---
    dt_grid = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        param_grid={"max_depth": [3, 4, 5, 6, 7, 8]},
        cv=5,
        scoring="r2",
    )
    dt_grid.fit(X_train, y_train)
    dt = dt_grid.best_estimator_

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)

    # --- Gradient Boosting ---
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)

    models = {"Linear Regression": lr, "Decision Tree (GridSearchCV)": dt,
              "Random Forest": rf, "Gradient Boosting": gb}

    all_metrics: dict[str, dict] = {}
    for name, mdl in models.items():
        y_pred = mdl.predict(X_test)
        all_metrics[name] = {
            "R²": r2_score(y_test, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "MAE": float(mean_absolute_error(y_test, y_pred)),
        }

    return models, all_metrics, (X_test, y_test)


# ──────────────────────────── plotting helpers ─────────────────────
def plot_correlation_heatmap(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)
    plt.close(fig)


def plot_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, col in zip(axes.flat, df.columns):
        sns.histplot(df[col], kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Distribution of {col}")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_feature_scatter(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, feat in zip(axes, FEATURES):
        ax.scatter(df[feat], df[TARGET], alpha=0.5, s=20, color="steelblue",
                   edgecolors="k", linewidths=0.3)
        ax.set_xlabel(feat)
        ax.set_ylabel(TARGET)
        ax.set_title(f"{feat} vs {TARGET}")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_feature_importance(model, model_name: str):
    if not hasattr(model, "feature_importances_"):
        st.info(f"{model_name} does not provide feature importances.")
        return
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar([FEATURES[i] for i in idx], importances[idx],
           color=["#e74c3c", "#3498db", "#2ecc71"], edgecolor="black")
    ax.set_ylabel("Importance")
    ax.set_title(f"Feature Importance — {model_name}")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_actual_vs_predicted(model, X_test, y_test, model_name: str):
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_test, y_pred, alpha=0.6, s=25, color="steelblue",
               edgecolors="k", linewidths=0.3)
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.set_title(f"Actual vs Predicted — {model_name}")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_model_comparison(all_metrics: dict):
    names = list(all_metrics.keys())
    r2_vals = [all_metrics[n]["R²"] for n in names]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
    bars = ax.barh(names, r2_vals, color=colors, edgecolor="black")
    ax.set_xlabel("R² Score")
    ax.set_title("Model Comparison (R²)")
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ──────────────────────────── main UI ──────────────────────────────
def main():
    st.set_page_config(
        page_title="Boston Housing Predictor",
        page_icon="🏠",
        layout="wide",
    )

    # ---- custom CSS for a clean look ----
    st.markdown("""
    <style>
    .block-container {padding-top: 1.5rem;}
    div[data-testid="stMetric"] {
        background: #f0f2f6; border-radius: .6rem; padding: .8rem;
    }
    </style>""", unsafe_allow_html=True)

    # ---- header ----
    st.title("🏠 Boston Housing Price Predictor")
    st.caption(
        "Trained on the same pipeline as **boston_housing_model.ipynb** — "
        "4 models, 3 features (RM, LSTAT, PTRATIO), target = MEDV."
    )

    # ---- sidebar: model picker ----
    st.sidebar.header("⚙️ Settings")
    selected_model_name = st.sidebar.selectbox("Select model", MODEL_NAMES, index=2)

    df = load_data()
    models, all_metrics, (X_test, y_test) = train_all_models()
    model = models[selected_model_name]
    metrics = all_metrics[selected_model_name]

    # ---- tabs ----
    tab_predict, tab_eda, tab_perf, tab_imp = st.tabs(
        ["🔮 Predict", "📊 Data Explorer", "📈 Model Performance", "🧩 Feature Importance"]
    )

    # ======================= PREDICT TAB ==========================
    with tab_predict:
        st.subheader("Enter house details")
        c1, c2, c3 = st.columns(3)
        rm = c1.slider(
            "RM — Avg. rooms",
            min_value=float(df["RM"].min()),
            max_value=float(df["RM"].max()),
            value=6.0, step=0.05,
        )
        lstat = c2.slider(
            "LSTAT — % lower-status",
            min_value=float(df["LSTAT"].min()),
            max_value=float(df["LSTAT"].max()),
            value=10.0, step=0.1,
        )
        ptratio = c3.slider(
            "PTRATIO — Pupil-teacher ratio",
            min_value=float(df["PTRATIO"].min()),
            max_value=float(df["PTRATIO"].max()),
            value=15.0, step=0.1,
        )

        input_df = pd.DataFrame([{"RM": rm, "LSTAT": lstat, "PTRATIO": ptratio}])
        prediction = model.predict(input_df)[0]

        st.divider()
        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.metric("💰 Predicted Price", f"${prediction:,.0f}")
            st.caption(f"Model: **{selected_model_name}**")
        with col_right:
            st.markdown("##### Your inputs")
            st.dataframe(input_df, hide_index=True, use_container_width=True)

    # ======================= EDA TAB ==============================
    with tab_eda:
        st.subheader("Dataset overview")
        st.dataframe(df.describe(), use_container_width=True)

        st.subheader("Feature distributions")
        plot_distributions(df)

        st.subheader("Scatter: features vs price")
        plot_feature_scatter(df)

        st.subheader("Correlation heatmap")
        plot_correlation_heatmap(df)

    # ======================= PERFORMANCE TAB ======================
    with tab_perf:
        st.subheader(f"Metrics for **{selected_model_name}**")
        m1, m2, m3 = st.columns(3)
        m1.metric("R²", f"{metrics['R²']:.4f}")
        m2.metric("RMSE", f"${metrics['RMSE']:,.0f}")
        m3.metric("MAE", f"${metrics['MAE']:,.0f}")

        st.subheader("Actual vs Predicted")
        plot_actual_vs_predicted(model, X_test, y_test, selected_model_name)

        st.subheader("All models — R² comparison")
        plot_model_comparison(all_metrics)

        st.subheader("Full metrics table")
        metrics_df = pd.DataFrame(all_metrics).T
        metrics_df["RMSE"] = metrics_df["RMSE"].map(lambda x: f"${x:,.0f}")
        metrics_df["MAE"] = metrics_df["MAE"].map(lambda x: f"${x:,.0f}")
        metrics_df["R²"] = metrics_df["R²"].map(lambda x: f"{x:.4f}")
        st.dataframe(metrics_df, use_container_width=True)

    # ======================= FEATURE IMPORTANCE TAB ===============
    with tab_imp:
        st.subheader(f"Feature Importance — {selected_model_name}")
        plot_feature_importance(model, selected_model_name)

        st.divider()
        st.markdown("**Feature descriptions**")
        for feat, desc in FEATURE_DESCRIPTIONS.items():
            st.markdown(f"- **{feat}**: {desc}")


if __name__ == "__main__":
    main()
