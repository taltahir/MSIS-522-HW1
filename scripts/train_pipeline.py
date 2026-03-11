#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from sklearn.neural_network import MLPRegressor

import joblib

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - handled in app runtime
    LGBMRegressor = None

try:
    import shap
except Exception:  # pragma: no cover - handled in app runtime
    shap = None

from config import (
    DATA_PATH,
    TARGET_COL,
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
    SAMPLE_SIZE,
    MAX_SHAP_SAMPLES,
    MAX_PAIRPLOT_SAMPLES,
    FEATURE_SUBSET_SIZES,
)

ARTIFACTS_DIR = Path("artifacts")
FIG_DIR = ARTIFACTS_DIR / "figures"
MODEL_DIR = ARTIFACTS_DIR / "models"
SHAP_DIR = ARTIFACTS_DIR / "shap"
PRED_DIR = ARTIFACTS_DIR / "predictions"
META_DIR = ARTIFACTS_DIR / "metadata"

UNIT_MAP = {
    "age": "years",
    "income": "USD (annual)",
    "daily_gaming_hours": "hours/day",
    "weekly_sessions": "sessions/week",
    "years_gaming": "years",
    "sleep_hours": "hours/night",
    "caffeine_intake": "units (unspecified)",
    "exercise_hours": "hours/week",
    "stress_level": "1-10 score",
    "anxiety_score": "0-10 score",
    "depression_score": "0-10 score",
    "social_interaction_score": "0-10 score",
    "relationship_satisfaction": "0-10 score",
    "academic_performance": "0-10 score",
    "work_productivity": "0-10 score",
    "addiction_level": "0-10 score",
    "multiplayer_ratio": "0-1 ratio",
    "toxic_exposure": "0-10 score",
    "violent_games_ratio": "0-1 ratio",
    "mobile_gaming_ratio": "0-1 ratio",
    "night_gaming_ratio": "0-1 ratio",
    "weekend_gaming_hours": "hours/weekend",
    "friends_gaming_count": "count",
    "online_friends": "count",
    "streaming_hours": "hours/week",
    "esports_interest": "0-10 score",
    "headset_usage": "0-1 indicator",
    "microtransactions_spending": "currency (unspecified)",
    "parental_supervision": "0-10 score",
    "loneliness_score": "0-10 score",
    "aggression_score": "0-10 score",
    "happiness_score": "0-10 score",
    "bmi": "kg/m^2",
    "screen_time_total": "hours/day",
    "eye_strain_score": "0-10 score",
    "back_pain_score": "0-10 score",
    "competitive_rank": "rank (1-100)",
    "internet_quality": "1-10 score",
}


def label_with_unit(feature: str) -> str:
    unit = UNIT_MAP.get(feature)
    return f"{feature} ({unit})" if unit else feature

# ---------- utilities ----------

def ensure_dirs() -> None:
    for d in [FIG_DIR, MODEL_DIR, SHAP_DIR, PRED_DIR, META_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
    return df


def count_rows(path: str) -> int:
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def clean_feature_names(names: List[str]) -> List[str]:
    cleaned = []
    for name in names:
        name = name.replace("num__", "").replace("cat__", "")
        cleaned.append(name)
    return cleaned


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def compute_cv_rmse(estimator, X, y) -> Tuple[float, float]:
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        estimator,
        X,
        y,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    rmses = np.sqrt(-scores)
    return float(rmses.mean()), float(rmses.std())


# ---------- EDA ----------

def generate_eda(df: pd.DataFrame, target: str) -> Dict[str, str]:
    insights: Dict[str, str] = {}
    sns.set_theme(style="whitegrid")

    # Target distribution
    plt.figure(figsize=(7, 4))
    sns.histplot(df[target], kde=True, bins=30, color="#3B82F6")
    plt.title(f"Distribution of {target}")
    plt.xlabel(label_with_unit(target))
    plt.ylabel("Count of participants")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "target_distribution.png", dpi=150)
    plt.close()

    skew = float(df[target].skew())
    q1, q3 = np.percentile(df[target], [25, 75])
    iqr = q3 - q1
    outlier_frac = float(((df[target] < q1 - 1.5 * iqr) | (df[target] > q3 + 1.5 * iqr)).mean())
    target_min = float(df[target].min())
    target_max = float(df[target].max())
    target_mean = float(df[target].mean())
    skew_desc = "right-skewed" if skew > 0.5 else "left-skewed" if skew < -0.5 else "roughly symmetric"
    insights["target_distribution"] = (
        f"The target distribution is {skew_desc} (skew={skew:.2f}). "
        f"Scores range from {target_min:.1f} to {target_max:.1f} (mean={target_mean:.2f}). "
        f"About {outlier_frac:.1%} of observations fall outside the IQR outlier bounds. "
        "We use RMSE and MAE to remain sensitive to wide tails while capturing overall fit."
    )

    # Sample for scatter-based plots
    sample_scatter = df.sample(min(10000, len(df)), random_state=RANDOM_STATE)

    # Gender vs target
    if "gender" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x="gender", y=target, data=df, hue="gender", palette="Set2", legend=False)
        plt.title(f"{target} by Gender")
        plt.xlabel("Gender")
        plt.ylabel(label_with_unit(target))
        # Overlay mean markers for clarity
        means_df = df.groupby("gender")[target].mean().reset_index()
        sns.pointplot(
            data=means_df,
            x="gender",
            y=target,
            color="black",
            markers="D",
            linestyles="",
            errorbar=None,
        )
        plt.tight_layout()
        plt.savefig(FIG_DIR / "gender_vs_target.png", dpi=150)
        plt.close()

        gender_means = df.groupby("gender")[target].mean().sort_values()
        low = gender_means.index[0]
        high = gender_means.index[-1]
        low_val = float(gender_means.iloc[0])
        high_val = float(gender_means.iloc[-1])
        insights["gender_vs_target"] = (
            f"Average {target} varies by gender, with {low} showing the lowest mean ({low_val:.2f}) "
            f"and {high} the highest ({high_val:.2f}). "
            "The spread suggests gender may contribute modestly to target variation. "
            "Differences are visible but not extreme, so gender likely interacts with other drivers."
        )

        # Mean difference from overall average (highlights subtle differences)
        overall_mean = float(df[target].mean())
        delta_df = means_df.copy()
        delta_df["delta"] = delta_df[target] - overall_mean
        plt.figure(figsize=(6, 4))
        sns.barplot(data=delta_df, x="gender", y="delta", hue="gender", palette="Set2", legend=False)
        plt.axhline(0, color="black", linewidth=1)
        plt.title(f"{target} Mean Difference vs Overall")
        plt.xlabel("Gender")
        plt.ylabel(f"Mean difference ({label_with_unit(target)})")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "gender_mean_delta.png", dpi=150)
        plt.close()

        insights["gender_mean_delta"] = (
            "This chart shows how each gender's mean differs from the overall average. "
            "Values close to zero indicate minimal differences, which is the case in this dataset."
        )

    # Daily gaming hours vs target (binned)
    if "daily_gaming_hours" in df.columns:
        bins = [0, 1, 2, 4, 6, 8, 12, df["daily_gaming_hours"].max() + 0.01]
        labels = ["<1h", "1-2h", "2-4h", "4-6h", "6-8h", "8-12h", "12h+"]
        df["gaming_hours_bin"] = pd.cut(df["daily_gaming_hours"], bins=bins, labels=labels, include_lowest=True)
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, x="gaming_hours_bin", y=target, color="#93C5FD")
        plt.title(f"{target} by Daily Gaming Hours (Binned)")
        plt.xlabel(label_with_unit("daily_gaming_hours"))
        plt.ylabel(label_with_unit(target))
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "gaming_hours_binned.png", dpi=150)
        plt.close()

        insights["gaming_hours_binned"] = (
            "This chart bins daily gaming time and shows the distribution of target scores within each band. "
            "If the boxes are similar across bins, it suggests daily gaming time has a weaker relationship with the target."
        )

        df = df.drop(columns=["gaming_hours_bin"])

    # Sleep hours vs target
    if "sleep_hours" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.regplot(x="sleep_hours", y=target, data=sample_scatter, scatter_kws={"alpha": 0.2, "s": 12})
        plt.title(f"{target} vs Sleep Hours")
        plt.xlabel(label_with_unit("sleep_hours"))
        plt.ylabel(label_with_unit(target))
        plt.tight_layout()
        plt.savefig(FIG_DIR / "sleep_hours_vs_target.png", dpi=150)
        plt.close()

        insights["sleep_hours_vs_target"] = (
            "Sleep duration often correlates with mental health indicators; this plot shows the overall trend. "
            "A downward slope would suggest higher sleep is linked with lower target scores."
        )

    # Stress level vs target
    if "stress_level" in df.columns:
        plt.figure(figsize=(7, 4))
        sns.boxplot(x="stress_level", y=target, data=sample_scatter, color="#F59E0B")
        plt.title(f"{target} by Stress Level")
        plt.xlabel("Stress level (higher = more stress)")
        plt.ylabel(label_with_unit(target))
        plt.tight_layout()
        plt.savefig(FIG_DIR / "stress_level_vs_target.png", dpi=150)
        plt.close()

        insights["stress_level_vs_target"] = (
            "Grouping by stress level highlights how the target changes at higher stress values. "
            "Wider boxes and higher medians indicate more variability and higher scores in those groups."
        )

    # Top correlated features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target]
    corr = df[numeric_cols + [target]].corr()[target].drop(target)
    top_features = corr.abs().sort_values(ascending=False).head(2).index.tolist()

    for idx, feat in enumerate(top_features, start=1):
        plt.figure(figsize=(6, 4))
        sns.regplot(x=feat, y=target, data=sample_scatter, scatter_kws={"alpha": 0.2, "s": 12})
        plt.title(f"{target} vs {feat}")
        plt.xlabel(label_with_unit(feat))
        plt.ylabel(label_with_unit(target))
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"feature_scatter_{idx}.png", dpi=150)
        plt.close()

        r = corr.loc[feat]
        direction = "positive" if r > 0 else "negative"
        insights[f"feature_scatter_{idx}"] = (
            f"{feat} shows a {direction} relationship with {target} (r={r:.2f}). "
            "The trend line highlights how changes in this feature align with target shifts. "
            "Scatter around the line hints at additional variables that modulate the relationship."
        )

    # Pairplot (sampled)
    pair_cols = [target] + top_features
    pair_sample = df[pair_cols].sample(min(MAX_PAIRPLOT_SAMPLES, len(df)), random_state=RANDOM_STATE)
    sns.pairplot(pair_sample, diag_kind="kde")
    plt.savefig(FIG_DIR / "pairplot.png", dpi=150)
    plt.close()

    insights["pairplot"] = (
        "The pair plot summarizes how the most correlated features jointly relate to the target. "
        "Clustering and slope patterns suggest potential multicollinearity and non-linear effects. "
        "These patterns motivate tree-based models that capture interactions."
    )

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr_matrix = df[numeric_cols + [target]].corr()
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    # Top correlations
    corr_pairs = (
        corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool))
        .stack()
        .reset_index()
    )
    corr_pairs.columns = ["feature_1", "feature_2", "corr"]
    corr_pairs = corr_pairs[corr_pairs["feature_1"] < corr_pairs["feature_2"]]
    top_pair = corr_pairs.iloc[corr_pairs["corr"].abs().idxmax()]
    insights["correlation_heatmap"] = (
        f"The strongest correlation appears between {top_pair['feature_1']} and {top_pair['feature_2']} "
        f"(r={top_pair['corr']:.2f}). This may indicate overlapping information that could affect linear models. "
        "We keep correlated features but compare regularized linear models to mitigate multicollinearity."
    )

    return insights


# ---------- modeling ----------

def build_preprocessors(categorical_features: List[str], numeric_features: List[str]):
    numeric_scaled = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    numeric_unscaled = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess_scaled = ColumnTransformer(
        transformers=[
            ("num", numeric_scaled, numeric_features),
            ("cat", categorical, categorical_features),
        ],
        remainder="drop",
    )

    preprocess_unscaled = ColumnTransformer(
        transformers=[
            ("num", numeric_unscaled, numeric_features),
            ("cat", categorical, categorical_features),
        ],
        remainder="drop",
    )

    return preprocess_scaled, preprocess_unscaled


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    preprocess_scaled,
    preprocess_unscaled,
) -> Tuple[pd.DataFrame, Dict[str, dict], Dict[str, str]]:
    results = []
    best_params: Dict[str, dict] = {}
    model_files: Dict[str, str] = {}
    target_label = y_test.name if getattr(y_test, "name", None) else "Target"
    target_label_with_unit = label_with_unit(target_label)

    model_specs = {
        "LinearRegression": {
            "estimator": LinearRegression(),
            "param_grid": None,
            "preprocess": preprocess_scaled,
            "use_grid": False,
        },
        "Lasso": {
            "estimator": Lasso(max_iter=5000, random_state=RANDOM_STATE),
            "param_grid": {"model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
            "preprocess": preprocess_scaled,
            "use_grid": True,
        },
        "Ridge": {
            "estimator": Ridge(random_state=RANDOM_STATE),
            "param_grid": {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
            "preprocess": preprocess_scaled,
            "use_grid": True,
        },
        "DecisionTree": {
            "estimator": DecisionTreeRegressor(random_state=RANDOM_STATE),
            "param_grid": {
                "model__max_depth": [3, 5, 7],
                "model__min_samples_leaf": [10, 20, 50],
            },
            "preprocess": preprocess_unscaled,
            "use_grid": True,
        },
        "RandomForest": {
            "estimator": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            "param_grid": {
                "model__n_estimators": [50, 100],
                "model__max_depth": [5, 8],
                "model__min_samples_leaf": [10, 20],
            },
            "preprocess": preprocess_unscaled,
            "use_grid": True,
        },
        "MLPRegressor": {
            "estimator": MLPRegressor(
                hidden_layer_sizes=(128, 128),
                activation="relu",
                solver="adam",
                max_iter=200,
                random_state=RANDOM_STATE,
            ),
            "param_grid": None,
            "preprocess": preprocess_scaled,
            "use_grid": False,
        },
    }

    if LGBMRegressor is not None:
        model_specs["LightGBM"] = {
            "estimator": LGBMRegressor(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "param_grid": {
                "model__n_estimators": [50, 100],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1],
                "model__num_leaves": [31, 63],
            },
            "preprocess": preprocess_unscaled,
            "use_grid": True,
        }

    for name, spec in model_specs.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", spec["preprocess"]),
                ("model", spec["estimator"]),
            ]
        )

        if spec["use_grid"] and spec["param_grid"]:
            search = GridSearchCV(
                pipeline,
                spec["param_grid"],
                cv=CV_FOLDS,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            best_estimator = search.best_estimator_
            best_params[name] = search.best_params_
        else:
            best_estimator = pipeline.fit(X_train, y_train)
            best_params[name] = {}

        cv_rmse_mean, cv_rmse_std = compute_cv_rmse(best_estimator, X_train, y_train)
        y_pred = best_estimator.predict(X_test)
        metrics = regression_metrics(y_test, y_pred)

        results.append(
            {
                "model": name,
                "cv_rmse_mean": cv_rmse_mean,
                "cv_rmse_std": cv_rmse_std,
                "test_rmse": metrics["rmse"],
                "test_mae": metrics["mae"],
                "test_r2": metrics["r2"],
            }
        )

        # Save model
        model_path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(best_estimator, model_path)
        model_files[name] = str(model_path)

        # Save predicted vs actual plot
        plt.figure(figsize=(5, 4))
        plt.scatter(y_test, y_pred, alpha=0.3)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
        plt.xlabel(f"Actual {target_label_with_unit}")
        plt.ylabel(f"Predicted {target_label_with_unit}")
        plt.title(f"Predicted vs Actual - {name}")
        plt.tight_layout()
        plt.savefig(PRED_DIR / f"{name}_pred_vs_actual.png", dpi=150)
        plt.close()

        if name == "MLPRegressor":
            mlp_model = best_estimator.named_steps["model"]
            if hasattr(mlp_model, "loss_curve_"):
                plt.figure(figsize=(6, 4))
                plt.plot(mlp_model.loss_curve_, color="#2563EB")
                plt.title("MLP Training Loss Curve")
                plt.xlabel("Epoch")
                plt.ylabel("Loss (MSE)")
                plt.tight_layout()
                plt.savefig(FIG_DIR / "mlp_loss_curve.png", dpi=150)
                plt.close()

    results_df = pd.DataFrame(results).sort_values("test_rmse")
    return results_df, best_params, model_files


def plot_model_comparison(metrics_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4))
    sns.barplot(data=metrics_df, x="model", y="test_rmse", hue="model", palette="Blues_r", legend=False)
    plt.xticks(rotation=30, ha="right")
    plt.title("Test RMSE by Model")
    plt.ylabel("RMSE (lower is better)")
    for idx, row in metrics_df.reset_index().iterrows():
        plt.text(idx, row["test_rmse"], f"{row['test_rmse']:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_comparison_rmse.png", dpi=150)
    plt.close()


def shap_analysis(best_tree_model_name: str, X_train, y_train) -> Dict[str, List[str]]:
    if shap is None:
        return {"top_features": []}

    model_path = MODEL_DIR / f"{best_tree_model_name}.joblib"
    pipeline = joblib.load(model_path)
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X_train)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    if X_transformed.shape[0] > MAX_SHAP_SAMPLES:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(X_transformed.shape[0], MAX_SHAP_SAMPLES, replace=False)
        X_sample = X_transformed[idx]
    else:
        X_sample = X_transformed

    feature_names = clean_feature_names(preprocessor.get_feature_names_out().tolist())

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Summary plot
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_summary.png", dpi=150)
    plt.close()

    # Bar plot
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_bar.png", dpi=150)
    plt.close()

    # Waterfall for max prediction
    preds = model.predict(X_sample)
    row_idx = int(np.argmax(preds))
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = np.array(base_val).reshape(-1)[0]
    explanation = shap.Explanation(
        values=shap_values[row_idx],
        base_values=base_val,
        data=X_sample[row_idx],
        feature_names=feature_names,
    )
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(explanation, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_waterfall.png", dpi=150)
    plt.close()

    # Aggregate importance and direction by original feature name
    mean_abs = np.abs(shap_values).mean(axis=0)
    aggregated_importance: Dict[str, float] = {}
    aggregated_direction: Dict[str, float] = {}
    for name, val, x_col, shap_col in zip(feature_names, mean_abs, X_sample.T, shap_values.T):
        base = "gender" if name.startswith("gender_") else name
        aggregated_importance[base] = aggregated_importance.get(base, 0.0) + float(val)

        if np.std(x_col) == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(x_col, shap_col)[0, 1])
        weighted = corr * float(val)
        aggregated_direction[base] = aggregated_direction.get(base, 0.0) + weighted

    ranked = sorted(aggregated_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = [k for k, _ in ranked][:10]

    shap_feature_summary = []
    for feat in top_features:
        score = aggregated_importance.get(feat, 0.0)
        direction_val = aggregated_direction.get(feat, 0.0)
        if direction_val > 0.0:
            direction = "Higher values increase prediction"
        elif direction_val < 0.0:
            direction = "Higher values decrease prediction"
        else:
            direction = "Mixed or neutral effect"
        shap_feature_summary.append(
            {
                "feature": feat,
                "mean_abs_shap": round(score, 6),
                "direction": direction,
            }
        )

    return {
        "top_features": top_features,
        "feature_names": feature_names,
        "feature_summary": shap_feature_summary,
    }


def feature_subset_performance(
    best_tree_model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_features: List[str],
    numeric_features: List[str],
    ranked_features: List[str],
) -> Dict[str, float]:
    if best_tree_model_name == "":
        return {}

    model_path = MODEL_DIR / f"{best_tree_model_name}.joblib"
    pipeline = joblib.load(model_path)
    base_model = pipeline.named_steps["model"]

    scores = {}
    for k in FEATURE_SUBSET_SIZES:
        k = min(k, len(ranked_features))
        subset = ranked_features[:k]
        subset_numeric = [f for f in subset if f in numeric_features]
        subset_categorical = [f for f in subset if f in categorical_features]

        preprocess_scaled, preprocess_unscaled = build_preprocessors(subset_categorical, subset_numeric)
        estimator = Pipeline(
            steps=[
                ("preprocess", preprocess_unscaled),
                ("model", clone(base_model)),
            ]
        )
        rmse_mean, rmse_std = compute_cv_rmse(estimator, X_train[subset], y_train)
        scores[str(k)] = {"rmse_mean": rmse_mean, "rmse_std": rmse_std, "features": subset}

    return scores


def plot_feature_subset_performance(scores: Dict[str, dict]) -> None:
    if not scores:
        return
    df = (
        pd.DataFrame.from_dict(scores, orient="index")
        .reset_index()
        .rename(columns={"index": "feature_count"})
    )
    df["feature_count"] = df["feature_count"].astype(int)
    df = df.sort_values("feature_count")

    plt.figure(figsize=(7, 4))
    plt.errorbar(
        df["feature_count"],
        df["rmse_mean"],
        yerr=df["rmse_std"],
        fmt="-o",
        color="#1D4ED8",
        ecolor="#93C5FD",
        capsize=4,
    )
    for _, row in df.iterrows():
        plt.text(row["feature_count"], row["rmse_mean"], f"{row['rmse_mean']:.3f}", ha="center", va="bottom", fontsize=8)
    plt.title("Cross-Validated RMSE vs Feature Count")
    plt.xlabel("Number of top-ranked features")
    plt.ylabel("RMSE (lower is better)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_subset_performance.png", dpi=150)
    plt.close()


def main() -> None:
    ensure_dirs()
    df = load_data()

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in data.")

    # Metadata
    categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [c for c in numeric_features if c != TARGET_COL]
    feature_types = {c: str(df[c].dtype) for c in df.columns if c != TARGET_COL}
    full_rows = count_rows(DATA_PATH)

    metadata = {
        "dataset_path": DATA_PATH,
        "full_n_rows": int(full_rows),
        "sample_n_rows": int(df.shape[0]),
        "sample_used": bool(SAMPLE_SIZE and full_rows > SAMPLE_SIZE),
        "n_rows": int(df.shape[0]),
        "n_features": int(df.shape[1] - 1),
        "target": TARGET_COL,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "feature_types": feature_types,
        "unit_map": UNIT_MAP,
    }

    insights = generate_eda(df, TARGET_COL)

    # Train/test split
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    metadata["feature_columns"] = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    preprocess_scaled, preprocess_unscaled = build_preprocessors(categorical_features, numeric_features)

    metrics_df, best_params, model_files = train_models(
        X_train,
        y_train,
        X_test,
        y_test,
        preprocess_scaled,
        preprocess_unscaled,
    )

    metrics_df.to_csv(META_DIR / "metrics.csv", index=False)
    save_json(best_params, META_DIR / "best_params.json")
    save_json(model_files, META_DIR / "model_files.json")

    # Model comparison plot
    plot_model_comparison(metrics_df)

    # Best model summary
    best_model_name = metrics_df.iloc[0]["model"]
    model_summary = {
        "best_model": best_model_name,
        "best_rmse": float(metrics_df.iloc[0]["test_rmse"]),
        "best_r2": float(metrics_df.iloc[0]["test_r2"]),
    }

    # Best tree model for SHAP
    tree_models = [m for m in metrics_df["model"].tolist() if m in ["DecisionTree", "RandomForest", "LightGBM"]]
    best_tree_model = ""
    if tree_models:
        tree_metrics = metrics_df[metrics_df["model"].isin(tree_models)].sort_values("test_rmse")
        best_tree_model = tree_metrics.iloc[0]["model"]

    model_summary["best_tree_model"] = best_tree_model

    shap_meta = {}
    if best_tree_model:
        shap_meta = shap_analysis(best_tree_model, X_train, y_train)
        model_summary["top_features"] = shap_meta.get("top_features", [])
        metadata["shap_feature_names"] = shap_meta.get("feature_names", [])
        metadata["shap_feature_summary"] = shap_meta.get("feature_summary", [])

        # Feature subset performance
        ranked_features = shap_meta.get("top_features", [])
        if ranked_features:
            remaining = [f for f in numeric_features + categorical_features if f not in ranked_features]
            ranked_features = ranked_features + remaining
        else:
            ranked_features = numeric_features + categorical_features
        subset_scores = feature_subset_performance(
            best_tree_model,
            X_train,
            y_train,
            categorical_features,
            numeric_features,
            ranked_features,
        )
        save_json(subset_scores, META_DIR / "feature_subset_scores.json")
        plot_feature_subset_performance(subset_scores)

    save_json(model_summary, META_DIR / "model_summary.json")

    # Feature stats for app inputs
    feature_stats = {}
    for col in numeric_features + [TARGET_COL]:
        series = df[col]
        feature_stats[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
        }
    for col in categorical_features:
        series = df[col]
        feature_stats[col] = {
            "mode": str(series.mode().iloc[0]),
            "unique": sorted(series.dropna().unique().tolist()),
        }

    save_json(feature_stats, META_DIR / "feature_stats.json")
    save_json(insights, META_DIR / "insights.json")
    save_json(metadata, META_DIR / "metadata.json")

    print("Training complete. Artifacts saved to ./artifacts")


if __name__ == "__main__":
    main()
