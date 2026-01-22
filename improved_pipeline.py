"""
Improved Water Quality Prediction Pipeline for EY AI & Data Challenge 2026
with advanced feature engineering, ensemble models, and hyperparameter tuning
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import joblib

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor

import xgboost as xgb
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ==================== PATHS ====================
NOTEBOOK_DIR = Path("/Users/user/Documents/GitHub/RiverIQ/notebooks")
MODEL_DIR = Path("/Users/user/Documents/GitHub/RiverIQ/models")
DATA_DIR = Path("/Users/user/Documents/GitHub/RiverIQ/data")

MODEL_DIR.mkdir(exist_ok=True, parents=True)

# ==================== DATA LOADING ====================
print("=" * 80)
print("LOADING DATASETS...")
print("=" * 80)

water_quality_df = pd.read_csv(NOTEBOOK_DIR / "water_quality_training_dataset.csv")
landsat_training_df = pd.read_csv(NOTEBOOK_DIR / "landsat_features_training.csv")
landsat_validation_df = pd.read_csv(NOTEBOOK_DIR / "landsat_features_validation.csv")
terraclimate_training_df = pd.read_csv(NOTEBOOK_DIR / "terraclimate_features_training.csv")
terraclimate_validation_df = pd.read_csv(NOTEBOOK_DIR / "terraclimate_features_validation.csv")
submission_df = pd.read_csv(DATA_DIR / "submission_template.csv")

print(f"✓ Water Quality: {water_quality_df.shape}")
print(f"✓ Landsat Training: {landsat_training_df.shape}")
print(f"✓ TerraClimate Training: {terraclimate_training_df.shape}")
print(f"✓ Submission Template: {submission_df.shape}")

# ==================== DATA PREPROCESSING ====================
print("\n" + "=" * 80)
print("DATA PREPROCESSING...")
print("=" * 80)

def clean_columns(df):
    """Standardize column names"""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df

def convert_dates(df):
    """Convert date columns to datetime"""
    if "sample_date" in df.columns:
        df["sample_date"] = pd.to_datetime(df["sample_date"], format="%d-%m-%Y", errors='coerce')
    return df

# Apply cleaning and date conversion
for df in [water_quality_df, landsat_training_df, landsat_validation_df, 
           terraclimate_training_df, terraclimate_validation_df, submission_df]:
    clean_columns(df)
    convert_dates(df)

print("✓ Columns cleaned and standardized")
print("✓ Dates converted to datetime")

# ==================== FEATURE ENGINEERING ====================
print("\n" + "=" * 80)
print("ADVANCED FEATURE ENGINEERING...")
print("=" * 80)

def engineer_features(df, is_training=True):
    """
    Comprehensive feature engineering pipeline
    """
    df = df.copy()
    
    # -------- Temporal Features --------
    df["year"] = df["sample_date"].dt.year
    df["month"] = df["sample_date"].dt.month
    df["day_of_year"] = df["sample_date"].dt.dayofyear
    df["quarter"] = df["month"].apply(lambda x: (x - 1) // 3 + 1)
    df["is_wet_season"] = df["month"].isin([10, 11, 12, 1, 2, 3]).astype(int)
    df["days_since_season_start"] = df["day_of_year"] % (365 // 2)
    
    # -------- Spatial Features --------
    df["lat_rad"] = np.deg2rad(df["latitude"])
    df["lon_rad"] = np.deg2rad(df["longitude"])
    df["lat_sin"] = np.sin(df["lat_rad"])
    df["lat_cos"] = np.cos(df["lat_rad"])
    df["lon_sin"] = np.sin(df["lon_rad"])
    df["lon_cos"] = np.cos(df["lon_rad"])
    df["latitude_squared"] = df["latitude"] ** 2
    df["longitude_squared"] = df["longitude"] ** 2
    df["latitude_longitude_interaction"] = df["latitude"] * df["longitude"]
    
    # -------- Spectral Indices --------
    # NDVI (Normalized Difference Vegetation Index)
    df["ndvi"] = (df["nir"] - df["green"]) / (df["nir"] + df["green"] + 1e-8)
    
    # EVI (Enhanced Vegetation Index)
    df["evi"] = 2.5 * (df["nir"] - df["green"]) / (df["nir"] + 6 * df["green"] + 1e-8)
    
    # BSI (Bare Soil Index)
    df["bsi"] = ((df["swir22"] + df["green"]) - (df["nir"] + df["green"])) / \
                ((df["swir22"] + df["green"]) + (df["nir"] + df["green"]) + 1e-8)
    
    # LSWI (Land Surface Water Index)
    df["lswi"] = (df["nir"] - df["swir16"]) / (df["nir"] + df["swir16"] + 1e-8)
    
    # -------- Spectral Ratios --------
    df["nir_swir16_ratio"] = df["nir"] / (df["swir16"] + 1e-8)
    df["green_swir22_ratio"] = df["green"] / (df["swir22"] + 1e-8)
    df["swir16_swir22_ratio"] = df["swir16"] / (df["swir22"] + 1e-8)
    df["nir_green_ratio"] = df["nir"] / (df["green"] + 1e-8)
    
    # -------- Spectral Indices Interactions --------
    df["ndmi_ndvi"] = df["ndmi"] * df["ndvi"]
    df["mndwi_ndvi"] = df["mndwi"] * df["ndvi"]
    df["ndvi_squared"] = df["ndvi"] ** 2
    df["mndwi_squared"] = df["mndwi"] ** 2
    df["ndmi_squared"] = df["ndmi"] ** 2
    
    # -------- Climate-Spectral Interactions --------
    df["pet_ndmi"] = df["pet"] * df["ndmi"]
    df["pet_mndwi"] = df["pet"] * df["mndwi"]
    df["pet_ndvi"] = df["pet"] * df["ndvi"]
    
    # -------- Data Quality Flags --------
    landsat_cols = ["nir", "green", "swir16", "swir22", "ndmi", "mndwi"]
    df["landsat_missing"] = df[landsat_cols].isna().any(axis=1).astype(int)
    
    # -------- Polynomial Features for Climate --------
    df["pet_squared"] = df["pet"] ** 2
    
    return df

# Engineer features for training data
df_train = water_quality_df.merge(
    landsat_training_df,
    on=["latitude", "longitude", "sample_date"],
    how="left"
).merge(
    terraclimate_training_df,
    on=["latitude", "longitude", "sample_date"],
    how="left"
)

df_train = engineer_features(df_train, is_training=True)
print(f"✓ Training data engineered: {df_train.shape}")

# Engineer features for validation data
df_val = submission_df.merge(
    landsat_validation_df,
    on=["latitude", "longitude", "sample_date"],
    how="left"
).merge(
    terraclimate_validation_df,
    on=["latitude", "longitude", "sample_date"],
    how="left"
)

df_val = engineer_features(df_val, is_training=False)
print(f"✓ Validation data engineered: {df_val.shape}")

# ==================== FEATURE SELECTION ====================
feature_cols = [
    # Temporal
    "year", "month", "day_of_year", "quarter", "is_wet_season", "days_since_season_start",
    # Spatial
    "lat_sin", "lat_cos", "lon_sin", "lon_cos",
    "latitude_squared", "longitude_squared", "latitude_longitude_interaction",
    # Landsat
    "nir", "green", "swir16", "swir22",
    "ndmi", "mndwi",
    # Spectral Indices
    "ndvi", "evi", "bsi", "lswi",
    # Spectral Ratios
    "nir_swir16_ratio", "green_swir22_ratio", "swir16_swir22_ratio", "nir_green_ratio",
    # Climate
    "pet", "pet_squared",
    # Interactions
    "pet_ndmi", "pet_mndwi", "pet_ndvi",
    "ndmi_ndvi", "mndwi_ndvi", "ndvi_squared", "mndwi_squared", "ndmi_squared",
    # Flags
    "landsat_missing"
]

# Keep only columns that exist
feature_cols = [col for col in feature_cols if col in df_train.columns]
print(f"✓ Selected {len(feature_cols)} features")

# ==================== DATA PREPARATION ====================
print("\n" + "=" * 80)
print("DATA PREPARATION...")
print("=" * 80)

targets = [
    "total_alkalinity",
    "electrical_conductance",
    "dissolved_reactive_phosphorus"
]

X_train = df_train[feature_cols]
X_val = df_val[feature_cols]

# Split training data for validation during model development
train_split = int(0.8 * len(X_train))
X_train_split = X_train.iloc[:train_split]
X_dev = X_train.iloc[train_split:]

y_train_split = {target: df_train[target].iloc[:train_split] for target in targets}
y_dev = {target: df_train[target].iloc[train_split:] for target in targets}

# Advanced imputation
print("Imputing missing values with KNN...")
imputer = KNNImputer(n_neighbors=5)

X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train_split),
    columns=feature_cols
)

X_dev_imputed = pd.DataFrame(
    imputer.transform(X_dev),
    columns=feature_cols
)

X_val_imputed = pd.DataFrame(
    imputer.transform(X_val),
    columns=feature_cols
)

X_full_imputed = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=feature_cols
)

print(f"✓ Training data imputed: {X_train_imputed.shape}")
print(f"✓ Dev data imputed: {X_dev_imputed.shape}")
print(f"✓ Validation data imputed: {X_val_imputed.shape}")

# ==================== MODEL EVALUATION ====================
def evaluate_models(y_true, y_pred, target_name):
    """Evaluate model performance"""
    return {
        "target": target_name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MAPE": np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    }

# ==================== MODEL TRAINING ====================
print("\n" + "=" * 80)
print("TRAINING ENSEMBLE MODELS...")
print("=" * 80)

all_results = []
final_models = {}

for target in targets:
    print(f"\n--- Training models for {target} ---")
    
    y_train_t = y_train_split[target]
    y_dev_t = y_dev[target]
    y_full_t = df_train[target]
    
    # -------- XGBoost --------
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    xgb_model.fit(
        X_train_imputed, y_train_t,
        eval_set=[(X_dev_imputed, y_dev_t)],
        callbacks=[xgb.callback.EarlyStopping(rounds=50)],
        verbose=False
    )
    
    xgb_pred = xgb_model.predict(X_dev_imputed)
    all_results.append(
        evaluate_models(y_dev_t, xgb_pred, target) | {"model": "XGBoost"}
    )
    print(f"  XGBoost R²: {r2_score(y_dev_t, xgb_pred):.4f}")
    
    # -------- LightGBM --------
    lgb_model = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    
    lgb_model.fit(
        X_train_imputed, y_train_t,
        eval_set=[(X_dev_imputed, y_dev_t)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=0)]
    )
    
    lgb_pred = lgb_model.predict(X_dev_imputed)
    all_results.append(
        evaluate_models(y_dev_t, lgb_pred, target) | {"model": "LightGBM"}
    )
    print(f"  LightGBM R²: {r2_score(y_dev_t, lgb_pred):.4f}")
    
    # -------- RandomForest --------
    rf_model = RandomForestRegressor(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_imputed, y_train_t)
    rf_pred = rf_model.predict(X_dev_imputed)
    all_results.append(
        evaluate_models(y_dev_t, rf_pred, target) | {"model": "RandomForest"}
    )
    print(f"  RandomForest R²: {r2_score(y_dev_t, rf_pred):.4f}")
    
    # -------- Gradient Boosting --------
    gb_model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    
    gb_model.fit(X_train_imputed, y_train_t)
    gb_pred = gb_model.predict(X_dev_imputed)
    all_results.append(
        evaluate_models(y_dev_t, gb_pred, target) | {"model": "GradientBoosting"}
    )
    print(f"  GradientBoosting R²: {r2_score(y_dev_t, gb_pred):.4f}")
    
    # -------- Ensemble (Voting Regressor) --------
    ensemble = VotingRegressor(
        estimators=[
            ('xgb', xgb_model),
            ('lgb', lgb_model),
            ('rf', rf_model),
            ('gb', gb_model)
        ],
        weights=[0.35, 0.35, 0.15, 0.15]
    )
    
    ensemble.fit(X_train_imputed, y_train_t)
    ensemble_pred = ensemble.predict(X_dev_imputed)
    all_results.append(
        evaluate_models(y_dev_t, ensemble_pred, target) | {"model": "Ensemble"}
    )
    print(f"  Ensemble R²: {r2_score(y_dev_t, ensemble_pred):.4f}")
    
    # Train final ensemble on full training data
    print(f"  Training final ensemble on full dataset...")
    
    final_xgb = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    final_lgb = lgb.LGBMRegressor(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    
    final_rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    final_gb = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    
    final_ensemble = VotingRegressor(
        estimators=[
            ('xgb', final_xgb),
            ('lgb', final_lgb),
            ('rf', final_rf),
            ('gb', final_gb)
        ],
        weights=[0.35, 0.35, 0.15, 0.15]
    )
    
    final_ensemble.fit(X_full_imputed, y_full_t)
    final_models[target] = final_ensemble
    
    # Save model
    joblib.dump(final_ensemble, MODEL_DIR / f"ensemble_{target}.pkl")
    print(f"  ✓ Final ensemble model saved")

# ==================== RESULTS SUMMARY ====================
print("\n" + "=" * 80)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 80)

results_df = pd.DataFrame(all_results)
print("\n" + results_df.to_string())

print("\n" + "=" * 80)
print("BEST MODELS PER TARGET")
print("=" * 80)

for target in targets:
    best_row = results_df[results_df["target"] == target].loc[
        results_df[results_df["target"] == target]["R2"].idxmax()
    ]
    print(f"\n{target.replace('_', ' ').title()}")
    print(f"  Model: {best_row['model']}")
    print(f"  R²: {best_row['R2']:.4f}")
    print(f"  RMSE: {best_row['RMSE']:.4f}")
    print(f"  MAE: {best_row['MAE']:.4f}")

# ==================== GENERATE SUBMISSION ====================
print("\n" + "=" * 80)
print("GENERATING SUBMISSION...")
print("=" * 80)

for target in targets:
    df_val[target] = final_models[target].predict(X_val_imputed)
    print(f"✓ Predictions made for {target}")

submission_output = df_val[["latitude", "longitude", "sample_date"] + targets].copy()
submission_output["sample_date"] = submission_output["sample_date"].dt.strftime("%d-%m-%Y")

output_path = DATA_DIR / "RiverIQ_submission_improved.csv"
submission_output.to_csv(output_path, index=False)

print(f"\n✓ Submission saved to: {output_path}")
print(f"  Rows: {len(submission_output)}")
print(f"  Columns: {list(submission_output.columns)}")

# Display sample
print("\nSample predictions:")
print(submission_output.head())

print("\n" + "=" * 80)
print("PIPELINE COMPLETE!")
print("=" * 80)
