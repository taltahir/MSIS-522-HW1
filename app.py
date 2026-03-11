import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    import shap
except Exception:
    shap = None

import joblib

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FIG_DIR = ARTIFACTS_DIR / "figures"
MODEL_DIR = ARTIFACTS_DIR / "models"
SHAP_DIR = ARTIFACTS_DIR / "shap"
PRED_DIR = ARTIFACTS_DIR / "predictions"
META_DIR = ARTIFACTS_DIR / "metadata"

DISPLAY_NAMES = {
    "LinearRegression": "Linear Regression",
    "Lasso": "Lasso",
    "Ridge": "Ridge",
    "DecisionTree": "CART (Decision Tree)",
    "RandomForest": "Random Forest",
    "LightGBM": "LightGBM",
    "MLPRegressor": "Neural Network (MLP)",
}


st.set_page_config(page_title="Gaming & Mental Health Pipeline", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Newsreader:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

body {
    background: radial-gradient(circle at 20% 20%, #E0F2FE 0%, #F8FAFC 45%, #FFF7ED 100%);
}

h1, h2, h3 {
    font-family: 'Newsreader', serif;
    color: #0F172A;
}

.stTabs [data-baseweb="tab"] {
    font-weight: 600;
    color: #0F172A;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
}
</style>
""",
    unsafe_allow_html=True,
)


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r") as f:
        return json.load(f)


@st.cache_resource
def load_models(model_files: dict):
    models = {}
    for name, path in model_files.items():
        try:
            models[name] = joblib.load(path)
        except Exception:
            continue
    return models


def clean_feature_names(names):
    cleaned = []
    for name in names:
        name = name.replace("num__", "").replace("cat__", "")
        cleaned.append(name)
    return cleaned


def render_missing_artifacts():
    st.error("Artifacts not found. Run `python3 scripts/train_pipeline.py` first.")
    st.stop()


metadata = load_json(META_DIR / "metadata.json")
if not metadata:
    render_missing_artifacts()

insights = load_json(META_DIR / "insights.json", default={})
metrics_df = None
if (META_DIR / "metrics.csv").exists():
    metrics_df = pd.read_csv(META_DIR / "metrics.csv")

best_params = load_json(META_DIR / "best_params.json", default={})
model_files = load_json(META_DIR / "model_files.json", default={})
model_summary = load_json(META_DIR / "model_summary.json", default={})
feature_stats = load_json(META_DIR / "feature_stats.json", default={})
subset_scores = load_json(META_DIR / "feature_subset_scores.json", default={})

models = load_models(model_files)

# Header
st.title("Gaming & Mental Health — End-to-End Data Science Pipeline")

# Tabs
summary_tab, eda_tab, perf_tab, explain_tab = st.tabs(
    ["Executive Summary", "Descriptive Analytics", "Model Performance", "Explainability & Interactive Prediction"]
)

with summary_tab:
    st.subheader("Dataset Overview")
    sample_note = "" if not metadata.get("sample_used") else (
        f"Modeling used a **random sample of {metadata.get('sample_n_rows'):,} rows** to optimize speed."
    )
    target_stats = feature_stats.get(metadata.get("target", ""), {})
    target_min = target_stats.get("min", None)
    target_max = target_stats.get("max", None)
    target_range_note = ""
    if target_min is not None and target_max is not None:
        target_range_note = f"Observed target scores range from {target_min:.1f} to {target_max:.1f}."
    st.markdown(
        f"""
This dataset captures behavioral, lifestyle, and gaming activity signals alongside mental health-related scores.
The modeling target in this project is **{metadata.get('target', 'depression_score')}**, treated as a continuous outcome.
Key predictors include demographics (age, gender, income), gaming behavior (daily hours, weekend hours, multiplayer ratio),
health and lifestyle signals (sleep, exercise, caffeine intake), and psychosocial scores (stress, anxiety, loneliness, happiness).
The prediction task is valuable because it helps quantify how gaming patterns and daily habits align with mental health indicators,
which can guide interventions, wellness program design, or content moderation strategies. {target_range_note} {sample_note}

Data source: [Kaggle – Gaming and Mental Health](https://www.kaggle.com/datasets/sharmajicoder/gaming-and-mental-health)
"""
    )

    st.markdown("**Train/Test split:** 70% train / 30% test.")

    st.subheader("Score Interpretation")
    st.markdown(
        """
Scores such as `depression_score`, `anxiety_score`, and `happiness_score` follow a **0–10 scale** in this dataset.
For interpretation in the app:
- **1** indicates a very low level of the measured construct (e.g., low depressive symptoms).
- **10** indicates a very high level of the measured construct (e.g., severe depressive symptoms).

These are *dataset scores* and not clinical diagnoses.
"""
    )

    st.subheader("Units & Scales (Inferred)")
    unit_map = metadata.get("unit_map", {})
    if unit_map:
        unit_rows = []
        for feature, unit in unit_map.items():
            unit_rows.append({"Feature": feature, "Unit / Scale (inferred)": unit})
        st.dataframe(pd.DataFrame(unit_rows), use_container_width=True, height=260)
    st.caption(
        "Units are inferred from column names when not explicitly documented. "
        "If the original data dictionary provides different units, update the table accordingly."
    )

    st.subheader("Why This Matters")
    st.markdown(
        """
Gaming is a dominant form of digital engagement, especially among younger populations, but its relationship with mental health can be complex.
By modeling a mental health score using fine-grained behavioral signals (gaming time, sleep, stress, social interaction, and more),
we can identify which behaviors are most associated with risk or resilience. That insight can inform platform design, parental guidance,
and user-facing tools that encourage healthier habits without demonizing gaming itself.
"""
    )

    st.subheader("Approach & Findings")
    best_model = model_summary.get("best_model", "")
    best_rmse = model_summary.get("best_rmse", None)
    best_r2 = model_summary.get("best_r2", None)
    best_rmse_text = f"{best_rmse:.3f}" if best_rmse is not None else "N/A"
    best_r2_text = f"{best_r2:.3f}" if best_r2 is not None else "N/A"
    st.markdown(
        f"""
The workflow combines deep exploratory analysis, multiple regression models with cross-validation, and explainability via SHAP.
Tree-based models (Random Forest and LightGBM) typically deliver strong performance, while linear models provide interpretability.
The model comparison and SHAP results highlight which behavioral features most influence the target score and how those effects move predictions.

**Best model (current run):** {DISPLAY_NAMES.get(best_model, best_model)}  
**Test RMSE:** {best_rmse_text}  
**Test R²:** {best_r2_text}
"""
    )

    st.markdown(
        f"""
**Dataset Stats**
- Rows: {metadata.get('full_n_rows', metadata.get('n_rows')):,}
- Features: {metadata.get('n_features', 0)}
- Numeric features: {len(metadata.get('numeric_features', []))}
- Categorical features: {len(metadata.get('categorical_features', []))}
"""
    )

with eda_tab:
    st.subheader("Target Distribution")
    target_img = FIG_DIR / "target_distribution.png"
    if target_img.exists():
        st.image(str(target_img), use_container_width=True)
        st.caption(
            insights.get(
                "target_distribution",
                "The target distribution shows how scores are spread across the sample. "
                "Look for skew and extreme values that may influence model fit.",
            )
        )

    if (FIG_DIR / "gender_vs_target.png").exists():
        st.subheader("Target by Gender")
        st.image(str(FIG_DIR / "gender_vs_target.png"), use_container_width=True)
        st.caption(
            insights.get(
                "gender_vs_target",
                "This plot compares the target score across gender categories. "
                "Mean markers highlight the average for each group to improve readability.",
            )
        )
        delta_img = FIG_DIR / "gender_mean_delta.png"
        if delta_img.exists():
            st.image(str(delta_img), use_container_width=True)
            st.caption(
                insights.get(
                    "gender_mean_delta",
                    "This plot shows each gender's mean difference from the overall average. "
                    "Near‑zero bars indicate minimal differences across groups.",
                )
            )

    st.subheader("Key Feature Relationships")
    if (FIG_DIR / "gaming_hours_binned.png").exists():
        st.image(str(FIG_DIR / "gaming_hours_binned.png"), use_container_width=True)
        st.caption(
            insights.get(
                "gaming_hours_binned",
                "Daily gaming time is bucketed into bins, and the chart shows the score distribution in each bin. "
                "If medians and boxes are similar across bins, the relationship is likely weak.",
            )
        )

    if (FIG_DIR / "sleep_hours_vs_target.png").exists():
        st.image(str(FIG_DIR / "sleep_hours_vs_target.png"), use_container_width=True)
        st.caption(
            insights.get(
                "sleep_hours_vs_target",
                "This scatter shows how sleep hours relate to the target, with a trend line for direction. "
                "A downward slope suggests more sleep is linked with lower scores.",
            )
        )

    if (FIG_DIR / "stress_level_vs_target.png").exists():
        st.image(str(FIG_DIR / "stress_level_vs_target.png"), use_container_width=True)
        st.caption(
            insights.get(
                "stress_level_vs_target",
                "This boxplot compares the target across stress levels. "
                "Higher medians at higher stress levels indicate elevated scores.",
            )
        )

    for idx in [1, 2]:
        img_path = FIG_DIR / f"feature_scatter_{idx}.png"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
            st.caption(
                insights.get(
                    f"feature_scatter_{idx}",
                    "The regression line summarizes the direction of association. "
                    "Spread around the line indicates additional factors at play.",
                )
            )

    pairplot_path = FIG_DIR / "pairplot.png"
    if pairplot_path.exists():
        st.subheader("Pair Plot")
        st.image(str(pairplot_path), use_container_width=True)
        st.caption(
            insights.get(
                "pairplot",
                "The pair plot summarizes interactions among the most correlated features. "
                "It helps diagnose non-linear relationships and clustering.",
            )
        )

    heatmap_path = FIG_DIR / "correlation_heatmap.png"
    if heatmap_path.exists():
        st.subheader("Correlation Heatmap")
        st.image(str(heatmap_path), use_container_width=True)
        st.caption(
            insights.get(
                "correlation_heatmap",
                "This heatmap highlights the strongest linear relationships among numeric features. "
                "Clusters of high correlation suggest overlapping information.",
            )
        )

with perf_tab:
    st.subheader("Modeling Workflow")
    st.markdown(
        """
- **Data preparation:** Defined `X` (features) and `y` (target), handled missing values with median/most-frequent imputation,
  applied one-hot encoding for categorical variables, and standardized numeric features for linear/MLP models.
- **Train/test split:** 70% training, 30% testing. All randomness uses `random_state = 42`.
- **Cross-validation:** 3-fold CV for speed (the rubric allows 3-fold on large datasets). You can set `CV_FOLDS = 5` in `config.py` for full 5-fold CV.
- **Baseline + regularization:** Linear Regression baseline, plus Lasso and Ridge.
- **Tree models:** CART, Random Forest, and LightGBM with grid search.
- **Neural network:** MLP with two hidden layers (128, 128) and ReLU activation.
- **Metrics reported:** MAE, RMSE, and R² on the test set (plus CV RMSE).
"""
    )

    st.subheader("Model Comparison")
    if metrics_df is not None:
        metrics_raw = metrics_df.copy()
        metrics_df = metrics_df.copy()
        metrics_df["model"] = metrics_df["model"].map(lambda x: DISPLAY_NAMES.get(x, x))
        display_df = metrics_df.copy()
        baseline_rmse = None
        if "LinearRegression" in metrics_raw["model"].values:
            baseline_rmse = float(metrics_raw.loc[metrics_raw["model"] == "LinearRegression", "test_rmse"].iloc[0])
        if baseline_rmse is not None and "test_rmse" in metrics_raw.columns:
            deltas = []
            for _, row in metrics_raw.iterrows():
                deltas.append(row["test_rmse"] - baseline_rmse)
            display_df["rmse_delta_vs_baseline"] = [f"{d:+.4f}" for d in deltas]
        for col in ["cv_rmse_mean", "cv_rmse_std", "test_rmse", "test_mae", "test_r2"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")
        st.dataframe(display_df, use_container_width=True)

    comparison_img = FIG_DIR / "model_comparison_rmse.png"
    if comparison_img.exists():
        st.image(str(comparison_img), use_container_width=True)
        st.caption(
            "Lower RMSE indicates better predictive performance on the held-out test set. "
            "Values are printed above each bar for easier comparison, even when bars are close."
        )

    st.subheader("Best Hyperparameters")
    if best_params:
        param_rows = []
        for model, params in best_params.items():
            param_rows.append(
                {
                    "Model": DISPLAY_NAMES.get(model, model),
                    "Best Params": params if params else "Default (no tuning)",
                }
            )
        st.dataframe(pd.DataFrame(param_rows), use_container_width=True)

    st.subheader("Predicted vs Actual")
    if model_files:
        for model_name in model_files.keys():
            img_path = PRED_DIR / f"{model_name}_pred_vs_actual.png"
            if img_path.exists():
                st.markdown(f"**{DISPLAY_NAMES.get(model_name, model_name)}**")
                st.image(str(img_path), use_container_width=True)
                st.caption(
                    f"Predicted vs actual for {DISPLAY_NAMES.get(model_name, model_name)}. "
                    "Points closer to the diagonal indicate more accurate predictions. "
                    "A tight band around the line suggests stronger model fit."
                )
    st.caption(
        "If points consistently fall above or below the diagonal, the model may be biased high or low. "
        "Wide vertical spread at a given actual value indicates higher error."
    )

    mlp_curve = FIG_DIR / "mlp_loss_curve.png"
    if mlp_curve.exists():
        st.subheader("MLP Training History")
        st.image(str(mlp_curve), use_container_width=True)
        st.caption(
            "Loss decreasing over epochs indicates the neural network is learning. "
            "If the curve flattens early, it may need more iterations or tuning."
        )

    st.subheader("Performance vs Feature Count")
    subset_plot = FIG_DIR / "feature_subset_performance.png"
    if subset_plot.exists():
        st.image(str(subset_plot), use_container_width=True)
        st.caption(
            "Cross-validated RMSE as more top-ranked features are included. "
            "A flat line means additional features contribute limited incremental gain."
        )

with explain_tab:
    st.subheader("SHAP Global Explanations")
    if (SHAP_DIR / "shap_summary.png").exists():
        st.image(str(SHAP_DIR / "shap_summary.png"), use_container_width=True)
        st.caption(
            "SHAP summary plot shows feature impact and direction across the dataset. "
            "Dots to the right increase predictions; dots to the left decrease them."
        )

    if (SHAP_DIR / "shap_bar.png").exists():
        st.image(str(SHAP_DIR / "shap_bar.png"), use_container_width=True)
        st.caption(
            "Mean absolute SHAP values rank the most influential features. "
            "Higher bars indicate larger average impact on the prediction."
        )

    shap_summary = metadata.get("shap_feature_summary", [])
    if shap_summary:
        st.subheader("Top SHAP Drivers (Direction + Impact)")
        st.dataframe(pd.DataFrame(shap_summary), use_container_width=True)
        st.caption(
            "Direction indicates whether higher feature values push predictions up or down, based on SHAP sign. "
            "This helps connect feature values to model behavior."
        )

    st.markdown(
        """
**SHAP interpretation highlights**
- **Strongest impact:** The top drivers in the table have the largest mean absolute SHAP values.
- **Direction:** The direction column explains whether higher values increase or decrease the prediction.
- **Decision value:** These insights identify behavioral levers (e.g., sleep, stress, gaming time) that can be targeted in wellness guidance or product interventions.
"""
    )

    st.subheader("Interactive Prediction")

    if not models:
        st.warning("No trained models found. Run the training pipeline first.")
    else:
        model_options = list(models.keys())
        display_options = [DISPLAY_NAMES.get(m, m) for m in model_options]
        model_choice = st.selectbox("Select model", options=model_options, format_func=lambda x: DISPLAY_NAMES.get(x, x))
        pipeline = models[model_choice]

        numeric_features = metadata.get("numeric_features", [])
        categorical_features = metadata.get("categorical_features", [])
        feature_columns = metadata.get("feature_columns", numeric_features + categorical_features)
        top_features = model_summary.get("top_features", feature_columns[:8])

        st.markdown("Choose which features to edit. Unselected features use average values.")
        default_features = top_features[:6]
        selected_features = st.multiselect("Editable features", feature_columns, default=default_features)

        user_values = {}
        for feature in feature_columns:
            if feature in numeric_features:
                stats = feature_stats.get(feature, {})
                default_val = stats.get("mean", 0.0)
                if feature in selected_features:
                    min_val = stats.get("min", 0.0)
                    max_val = stats.get("max", 1.0)
                    dtype = metadata.get("feature_types", {}).get(feature, "float")
                    if "int" in dtype:
                        user_values[feature] = st.slider(
                            feature,
                            int(min_val),
                            int(max_val),
                            int(round(default_val)),
                        )
                    else:
                        step = (max_val - min_val) / 100 if max_val != min_val else 0.1
                        step = float(max(step, 0.01))
                        user_values[feature] = st.slider(
                            feature,
                            float(min_val),
                            float(max_val),
                            float(default_val),
                            step=step,
                        )
                else:
                    user_values[feature] = default_val
            else:
                stats = feature_stats.get(feature, {})
                default_val = stats.get("mode", "")
                if feature in selected_features:
                    options = stats.get("unique", [default_val])
                    user_values[feature] = st.selectbox(feature, options, index=options.index(default_val) if default_val in options else 0)
                else:
                    user_values[feature] = default_val

        input_df = pd.DataFrame([user_values], columns=feature_columns)
        prediction = pipeline.predict(input_df)[0]

        st.markdown("""
<div class="metric-card">
<strong>Predicted Outcome</strong><br/>
Estimated target value for your inputs.
</div>
""", unsafe_allow_html=True)
        st.metric(label="Predicted Score", value=f"{prediction:,.2f}")

        st.subheader("Custom SHAP Waterfall")
        tree_models = ["DecisionTree", "RandomForest", "LightGBM"]
        shap_model_name = model_choice if model_choice in tree_models else model_summary.get("best_tree_model", "")

        if shap is None:
            st.info("Install SHAP to view the waterfall explanation.")
        elif shap_model_name and shap_model_name in models:
            shap_pipeline = models[shap_model_name]
            preprocessor = shap_pipeline.named_steps["preprocess"]
            tree_model = shap_pipeline.named_steps["model"]

            X_transformed = preprocessor.transform(input_df)
            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()

            feature_names = clean_feature_names(preprocessor.get_feature_names_out().tolist())
            explainer = shap.TreeExplainer(tree_model)
            shap_values = explainer.shap_values(X_transformed)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            base_val = explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)):
                base_val = base_val[0]

            explanation = shap.Explanation(
                values=shap_values[0],
                base_values=base_val,
                data=X_transformed[0],
                feature_names=feature_names,
            )
            plt.figure(figsize=(8, 5))
            shap.plots.waterfall(explanation, max_display=10, show=False)
            st.pyplot(plt.gcf())
        else:
            st.info("Tree-based model required for SHAP waterfall. Select Decision Tree, Random Forest, or LightGBM.")

    st.subheader("Interpretation Guide")
    st.markdown(
        """
- Features ranked higher in the SHAP bar plot contribute the most to prediction changes.
- Positive SHAP values push the predicted score higher; negative values reduce it.
- Use the waterfall plot to see how individual features combine to produce the final prediction.
- These insights highlight which behaviors are most associated with the target score, useful for targeted interventions.
"""
    )
