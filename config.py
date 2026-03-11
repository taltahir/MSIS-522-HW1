DATA_PATH = "gaming_mental_health_10M_40features.csv"

# Target: change here if you want a different outcome
TARGET_COL = "depression_score"

RANDOM_STATE = 42
TEST_SIZE = 0.30
CV_FOLDS = 3

# Use a sample for faster training; set to None for full dataset
SAMPLE_SIZE = 50_000

# SHAP and plotting samples for speed
MAX_SHAP_SAMPLES = 5000
MAX_PAIRPLOT_SAMPLES = 2000

# Feature subset sizes used for performance-vs-features curve
FEATURE_SUBSET_SIZES = [5, 10, 15, 20, 39]
