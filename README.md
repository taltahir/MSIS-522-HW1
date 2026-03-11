# Gaming & Mental Health — End-to-End Data Science Pipeline

This project builds a complete data science workflow on the **gaming_mental_health_10M_40features.csv** dataset:
exploratory analysis, model training with cross‑validation (Linear, Lasso, Ridge, CART, Random Forest, LightGBM, MLP),
SHAP explainability, and a Streamlit app.

## Quick Start

```bash
python3 -m pip install -r requirements.txt
python3 scripts/train_pipeline.py
streamlit run app.py
```

## Key Files

- `config.py` — dataset path, target column, sampling size, and CV settings
- `scripts/train_pipeline.py` — runs EDA, trains models, saves artifacts
- `artifacts/` — saved figures, models, SHAP plots, metrics
- `app.py` — Streamlit app with all required tabs

## Notes

- Default target: `depression_score`. Update `TARGET_COL` in `config.py` if needed.
- By default, training samples 50,000 rows for speed and uses 3-fold CV. Set `SAMPLE_SIZE = None` and `CV_FOLDS = 5` for the full HW run.
- All randomness uses `random_state = 42`.

## Deployment

Deploy using Streamlit Community Cloud or another service that supports `requirements.txt`.
Ensure `artifacts/` are generated before deployment.
