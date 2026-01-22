"""
Optimized Water Quality Prediction Pipeline - Simplified Version
Fast execution with proven effective models
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import joblib
from datetime import datetime

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ==================== PATHS ====================
NOTEBOOK_DIR = Path("/Users/user/Documents/GitHub/RiverIQ/notebooks")
DATA_DIR = Path("/Users/user/Documents/GitHub/RiverIQ/data")

# ==================== DATA LOADING ====================
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Loading datasets...")

water_quality_df = pd.read_csv(NOTEBOOK_DIR / "water_quality_training_dataset.csv")
landsat_training_df = pd.read_csv(NOTEBOOK_DIR / "landsat_features_training.csv")
terraclimate_training_df = pd.read_csv(NOTEBOOK_DIR / "terraclimate_features_training.csv")
submission_df = pd.read_csv(DATA_DIR / "submission_template.csv")
landsat_validation_df = pd.read_csv(NOTEBOOK_DIR / "landsat_features_validation.csv")
terraclimate_validation_df = pd.read_csv(NOTEBOOK_DIR / "terraclimate_features_validation.csv")

# ==================== PREPROCESSING ====================
def clean_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

def convert_dates(df):
    if "sample_date" in df.columns:
        df["sample_date"] = pd.to_datetime(df["sample_date"], format="%d-%m-%Y", errors='coerce')
    return df

# Clean all dataframes
for df in [water_quality_df, landsat_training_df, terraclimate_training_df, 
           submission_df, landsat_validation_df, terraclimate_validation_df]:
    clean_columns(df)
    convert_dates(df)

print(f"[{datetime.now().strftime('%H:%M:%S')}] Data loaded and cleaned")

# ==================== FEATURE ENGINEERING ====================
def engineer_features(df):
    """Advanced feature engineering"""
    df = df.copy()
    
    # Temporal
    df["year"] = df["sample_date"].dt.year
    df["month"] = df["sample_date"].dt.month
    df["day_of_year"] = df["sample_date"].dt.dayofyear
    df["quarter"] = ((df["month"] - 1) // 3) + 1
    df["is_wet_season"] = df["month"].isin([10, 11, 12, 1, 2, 3]).astype(int)
    
    # Spatial
    df["lat_rad"] = np.deg2rad(df["latitude"])
    df["lon_rad"] = np.deg2rad(df["longitude"])
    df["lat_sin"] = np.sin(df["lat_rad"])
    df["lat_cos"] = np.cos(df["lat_rad"])
    df["lon_sin"] = np.sin(df["lon_rad"])
    df["lon_cos"] = np.cos(df["lon_rad"])
    
    # Spectral Indices
    eps = 1e-8
    df["ndvi"] = (df["nir"] - df["green"]) / (df["nir"] + df["green"] + eps)
    df["evi"] = 2.5 * (df["nir"] - df["green"]) / (df["nir"] + 6*df["green"] + eps)
    df["bsi"] = ((df["swir22"] + df["green"]) - (df["nir"] + df["green"])) / ((df["swir22"] + df["green"]) + (df["nir"] + df["green"]) + eps)
    df["lswi"] = (df["nir"] - df["swir16"]) / (df["nir"] + df["swir16"] + eps)
    
    # Ratios
    df["nir_swir16_ratio"] = df["nir"] / (df["swir16"] + eps)
    df["green_swir22_ratio"] = df["green"] / (df["swir22"] + eps)
    df["nir_green_ratio"] = df["nir"] / (df["green"] + eps)
    
    # Interactions
    df["ndvi_ndmi"] = df["ndvi"] * df["ndmi"]
    df["ndvi_mndwi"] = df["ndvi"] * df["mndwi"]
    df["pet_ndvi"] = df["pet"] * df["ndvi"]
    df["pet_ndmi"] = df["pet"] * df["ndmi"]
    df["pet_mndwi"] = df["pet"] * df["mndwi"]
    
    # Polynomial
    df["ndvi_sq"] = df["ndvi"] ** 2
    df["ndmi_sq"] = df["ndmi"] ** 2
    df["mndwi_sq"] = df["mndwi"] ** 2
    df["pet_sq"] = df["pet"] ** 2
    
    # Spatial polynomial
    df["lat_sq"] = df["latitude"] ** 2
    df["lon_sq"] = df["longitude"] ** 2
    
    # Data quality
    landsat_cols = ["nir", "green", "swir16", "swir22", "ndmi", "mndwi"]
    df["landsat_missing"] = df[landsat_cols].isna().any(axis=1).astype(int)
    
    return df

print(f"[{datetime.now().strftime('%H:%M:%S')}] Engineering features...")

# Merge and engineer features
df_train = water_quality_df.merge(
    landsat_training_df, on=["latitude", "longitude", "sample_date"], how="left"
).merge(
    terraclimate_training_df, on=["latitude", "longitude", "sample_date"], how="left"
)

df_test = submission_df.merge(
    landsat_validation_df, on=["latitude", "longitude", "sample_date"], how="left"
).merge(
    terraclimate_validation_df, on=["latitude", "longitude", "sample_date"], how="left"
)

df_train = engineer_features(df_train)
df_test = engineer_features(df_test)

# Feature selection
feature_cols = [
    "year", "month", "day_of_year", "quarter", "is_wet_season",
    "lat_sin", "lat_cos", "lon_sin", "lon_cos", "lat_sq", "lon_sq",
    "nir", "green", "swir16", "swir22", "ndmi", "mndwi",
    "ndvi", "evi", "bsi", "lswi",
    "nir_swir16_ratio", "green_swir22_ratio", "nir_green_ratio",
    "ndvi_ndmi", "ndvi_mndwi", "pet_ndvi", "pet_ndmi", "pet_mndwi",
    "ndvi_sq", "ndmi_sq", "mndwi_sq", "pet_sq",
    "pet", "landsat_missing"
]

feature_cols = [col for col in feature_cols if col in df_train.columns]

X_train = df_train[feature_cols].copy()
X_test = df_test[feature_cols].copy()

# Imputation
print(f"[{datetime.now().strftime('%H:%M:%S')}] Imputing missing values...")
imputer = KNNImputer(n_neighbors=5)
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols)
X_test = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)

# Split for validation
train_idx = int(0.8 * len(X_train))
X_tr, X_val = X_train.iloc[:train_idx], X_train.iloc[train_idx:]

targets = ["total_alkalinity", "electrical_conductance", "dissolved_reactive_phosphorus"]
y_tr = {t: df_train[t].iloc[:train_idx] for t in targets}
y_val = {t: df_train[t].iloc[train_idx:] for t in targets}
y_full = {t: df_train[t] for t in targets}

# ==================== MODEL TRAINING ====================
print(f"[{datetime.now().strftime('%H:%M:%S')}] Training models...")

models = {}
results = []

for target in targets:
    print(f"\n  Training {target}...")
    
    # Base models
    rf = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=5, 
                               random_state=42, n_jobs=-1)
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                   subsample=0.8, random_state=42)
    lgb_model = lgb.LGBMRegressor(n_estimators=300, max_depth=7, learning_rate=0.05,
                                  num_leaves=31, subsample=0.8, random_state=42, verbosity=-1)
    
    # Ensemble - Voting
    ensemble = VotingRegressor(
        estimators=[('rf', rf), ('gb', gb), ('lgb', lgb_model)],
        weights=[0.3, 0.3, 0.4]
    )
    
    # Train on split data
    ensemble.fit(X_tr, y_tr[target])
    
    # Evaluate
    pred_val = ensemble.predict(X_val)
    r2 = r2_score(y_val[target], pred_val)
    rmse = np.sqrt(mean_squared_error(y_val[target], pred_val))
    mae = mean_absolute_error(y_val[target], pred_val)
    
    results.append({
        'Target': target,
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae
    })
    
    print(f"    R²: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    
    # Train final model on full data
    rf_final = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=5,
                                     random_state=42, n_jobs=-1)
    gb_final = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                         subsample=0.8, random_state=42)
    lgb_final = lgb.LGBMRegressor(n_estimators=300, max_depth=7, learning_rate=0.05,
                                  num_leaves=31, subsample=0.8, random_state=42, verbosity=-1)
    
    ensemble_final = VotingRegressor(
        estimators=[('rf', rf_final), ('gb', gb_final), ('lgb', lgb_final)],
        weights=[0.3, 0.3, 0.4]
    )
    
    ensemble_final.fit(X_train, y_full[target])
    models[target] = ensemble_final

# ==================== GENERATE SUBMISSION ====================
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Generating submission...")

submission = df_test[["latitude", "longitude", "sample_date"]].copy()
for target in targets:
    submission[target] = models[target].predict(X_test)

submission["sample_date"] = submission["sample_date"].dt.strftime("%d-%m-%Y")

output_path = DATA_DIR / "RiverIQ_submission_improved.csv"
submission.to_csv(output_path, index=False)

# ==================== RESULTS ====================
print(f"\n{'='*80}")
print("MODEL PERFORMANCE")
print(f"{'='*80}")
for r in results:
    print(f"{r['Target']:<30} R²={r['R2']:.4f}  RMSE={r['RMSE']:.2f}  MAE={r['MAE']:.2f}")

print(f"\n{'='*80}")
print(f"Submission saved to: {output_path}")
print(f"Rows: {len(submission)}")
print(f"\nSample predictions:")
print(submission.head())
print(f"{'='*80}")
