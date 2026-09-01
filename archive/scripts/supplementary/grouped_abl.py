#!/usr/bin/env python
# coding: utf-8
"""
grouped_ablation.py
===================
Grouped feature ablation to measure the Stage 1 (XGBoost wind/solar prediction)
contribution as a block, fixing the correlated-feature masking issue in
single-feature leave-one-out ablation.

Three configurations:
  BASELINE      – all 22 features
  NO_STAGE1     – remove Stage 1 outputs (Wind/Solar_Predicted_MW,
                  Wind/Solar_Forecast_Error_XGB); keep BMRS DA forecasts
  NO_WIND_AT_ALL – remove every wind/solar-related feature including BMRS DA

Replicates the exact data pipeline from 03_price_prediction_main.py /
05_additional_analysis.py. Same random seed (42), same XGB_PARAMS, same SVR_PARAMS.
Ensemble alpha is RE-OPTIMISED per configuration.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             precision_score, recall_score, f1_score, confusion_matrix)
from sklearn.svm import SVR
import xgboost as xgb
import holidays

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW  = os.path.join(BASE, 'data', 'raw')
DATA = os.path.join(BASE, 'data')

# Spike threshold from Section 5.4 (test-set 90th percentile)
SPIKE_THRESHOLD = 130.66

print("=" * 70)
print("GROUPED FEATURE ABLATION")
print("=" * 70)

# ================================================================
# 1. DATA LOADING (exact replica of 05_additional_analysis.py)
# ================================================================
print("\n[1] Loading data...")

df_prices  = pd.read_csv(os.path.join(RAW, 'elec_price_hourly_ember.csv'))
df_wind_p  = pd.read_csv(os.path.join(RAW, 'wind_gen_predicted_hourly_xgboost.csv'))
df_solar_p = pd.read_csv(os.path.join(RAW, 'solar_gen_predicted_hourly_xgboost.csv'))
df_demand  = pd.read_csv(os.path.join(RAW, 'elec_demand_outturn_hh_bmrs.csv'))
df_gas     = pd.read_csv(os.path.join(RAW, 'gas_sap_daily_icis.csv'))
df_co2     = pd.read_csv(os.path.join(RAW, 'carbon_ukets_futures_daily_investing.csv'))

def strip_tz(idx):
    if hasattr(idx, 'tz') and idx.tz is not None:
        return idx.tz_convert(None)
    return idx

# DA demand forecast
HAS_DA_DEMAND = False
path_demand_da = os.path.join(RAW, 'elec_demand_forecast_da_hh_bmrs.csv')
if os.path.exists(path_demand_da):
    df_demand_da = pd.read_csv(path_demand_da)
    df_demand_da['datetime'] = pd.to_datetime(df_demand_da['datetime'])
    df_demand_da.set_index('datetime', inplace=True)
    df_demand_da.index = strip_tz(df_demand_da.index)
    df_demand_da_hourly = df_demand_da[['Demand_DA_MW']].resample('h').mean().rename(
        columns={'Demand_DA_MW': 'Demand_MW'})
    da_days = (df_demand_da_hourly.index.max() - df_demand_da_hourly.index.min()).days
    if da_days >= 30:
        HAS_DA_DEMAND = True
        df_demand_hourly = df_demand_da_hourly
        print(f"  Using DA demand forecast ({da_days} days)")

if not HAS_DA_DEMAND:
    df_demand['Datetime'] = pd.to_datetime(df_demand['SETTLEMENT_DATE'], dayfirst=True, format='mixed') + \
                            pd.to_timedelta((df_demand['SETTLEMENT_PERIOD'] - 1) * 30, unit='m')
    df_demand.set_index('Datetime', inplace=True)
    df_demand_outturn_hourly = df_demand[['ND']].resample('h').mean().rename(columns={'ND': 'Demand_MW'})
    df_demand_hourly = df_demand_outturn_hourly.shift(24).dropna()
    print("  Using 24h persistence demand forecast")

# DA wind/solar forecasts
df_ws_da_hourly = pd.DataFrame()
path_ws_da = os.path.join(RAW, 'renew_gen_forecast_da_hourly_bmrs.csv')
if os.path.exists(path_ws_da):
    df_ws_da = pd.read_csv(path_ws_da)
    df_ws_da['datetime'] = pd.to_datetime(df_ws_da['datetime'])
    df_ws_da.set_index('datetime', inplace=True)
    df_ws_da.index = strip_tz(df_ws_da.index)
    ws_da_cols = [c for c in df_ws_da.columns if 'Forecast' in c]
    df_ws_da_hourly = df_ws_da[ws_da_cols].resample('h').mean()
    ws_da_days = (df_ws_da_hourly.index.max() - df_ws_da_hourly.index.min()).days
    if ws_da_days < 30:
        df_ws_da_hourly = pd.DataFrame()

# Wind
df_wind_p['timestamp'] = pd.to_datetime(df_wind_p['timestamp'])
df_wind_p.set_index(df_wind_p['timestamp'].dt.tz_convert(None), inplace=True)
df_wind_hourly = df_wind_p[['predicted_mw']].resample('h').mean().rename(
    columns={'predicted_mw': 'Wind_Predicted_MW'})

# Solar
df_solar_p['timestamp'] = pd.to_datetime(df_solar_p['timestamp'])
df_solar_p.set_index(df_solar_p['timestamp'].dt.tz_convert(None), inplace=True)
df_solar_hourly = df_solar_p[['predicted_mw']].resample('h').mean().rename(
    columns={'predicted_mw': 'Solar_Predicted_MW'})

# Prices
df_prices['Datetime'] = pd.to_datetime(df_prices['Datetime (UTC)'], dayfirst=True, format='mixed')
df_prices.set_index('Datetime', inplace=True)
df_prices_hourly = df_prices[['Price (EUR/MWhe)']].sort_index().rename(
    columns={'Price (EUR/MWhe)': 'Price_EUR'})

# Gas (1-day lag)
df_gas['Date'] = pd.to_datetime(df_gas['Date'], dayfirst=True, format='mixed')
df_gas = df_gas.sort_values('Date').set_index('Date')
gas_col = 'SAP actual day' if 'SAP actual day' in df_gas.columns else df_gas.columns[0]
df_gas_daily = df_gas[[gas_col]].rename(columns={gas_col: 'Gas_Price'})
df_gas_daily = df_gas_daily.shift(1)
df_gas_hourly = df_gas_daily.reindex(
    pd.date_range(df_gas.index.min(), df_gas.index.max(), freq='D')
).interpolate().resample('h').ffill()

# CO2 (1-day lag)
df_co2['Date'] = pd.to_datetime(df_co2['Date'], dayfirst=True, format='mixed')
df_co2 = df_co2.sort_values('Date').set_index('Date')
df_co2_daily = df_co2[['Price']].rename(columns={'Price': 'CO2_Price'})
df_co2_daily = df_co2_daily.shift(1)
df_co2_hourly = df_co2_daily.reindex(
    pd.date_range(df_co2.index.min(), df_co2.index.max(), freq='D')
).interpolate().resample('h').ffill()

# Merge
data = df_prices_hourly.join(df_demand_hourly, how='inner')
data = data.join(df_wind_hourly, how='inner')
data = data.join(df_solar_hourly, how='inner')
data = data.join(df_gas_hourly, how='inner')
data = data.join(df_co2_hourly, how='inner')

if not df_ws_da_hourly.empty:
    data = data.join(df_ws_da_hourly, how='left')
    for col in df_ws_da_hourly.columns:
        if col in data.columns:
            data[col] = data[col].ffill()

print(f"  Merged: {len(data):,} rows, {data.shape[1]} columns")
print(f"  Date range: {data.index.min()} -> {data.index.max()}")

# ================================================================
# 2. FEATURE ENGINEERING (exact replica)
# ================================================================
print("\n[2] Building features...")

data['Residual_Load'] = data['Demand_MW'] - (data['Wind_Predicted_MW'] + data['Solar_Predicted_MW'])
data['Theoretical_Cost'] = data['Gas_Price'] + (0.5 * data['CO2_Price'])
data['Cost_Load_Interaction'] = data['Theoretical_Cost'] * data['Residual_Load']
data['ResLoad_Roll_Mean_24'] = data['Residual_Load'].rolling(window=24, closed='left').mean()
data['ResLoad_Roll_Std_24']  = data['Residual_Load'].rolling(window=24, closed='left').std()

uk_holidays = holidays.UnitedKingdom(years=[2021, 2022, 2023, 2024, 2025])
data['is_holiday'] = data.index.map(lambda x: 1 if x in uk_holidays else 0)
data['hour_sin']   = np.sin(2 * np.pi * data.index.hour / 24)
data['hour_cos']   = np.cos(2 * np.pi * data.index.hour / 24)
data['is_weekend'] = data.index.dayofweek.isin([5, 6]).astype(int)

data['Price_Lag_24']       = data['Price_EUR'].shift(24)
data['Price_Lag_168']      = data['Price_EUR'].shift(168)
data['Price_Roll_Mean_24'] = data['Price_EUR'].rolling(window=24, closed='left').mean()

if 'Wind_Forecast_DA_MW' in data.columns:
    data['Wind_Forecast_Error_XGB'] = data['Wind_Predicted_MW'] - data['Wind_Forecast_DA_MW']
if 'Solar_Forecast_DA_MW' in data.columns:
    data['Solar_Forecast_Error_XGB'] = data['Solar_Predicted_MW'] - data['Solar_Forecast_DA_MW']
if 'Wind_Forecast_DA_MW' in data.columns and 'Solar_Forecast_DA_MW' in data.columns:
    data['Residual_Load_DA'] = data['Demand_MW'] - (data['Wind_Forecast_DA_MW'] + data['Solar_Forecast_DA_MW'])

data.dropna(inplace=True)

# Full feature list (same order as 05_additional_analysis.py)
ALL_FEATURES = [
    'Residual_Load', 'ResLoad_Roll_Mean_24', 'ResLoad_Roll_Std_24',
    'Gas_Price', 'CO2_Price', 'Theoretical_Cost', 'Cost_Load_Interaction',
    'Wind_Predicted_MW', 'Solar_Predicted_MW', 'Demand_MW',
    'Price_Lag_24', 'Price_Lag_168', 'Price_Roll_Mean_24',
    'hour_sin', 'hour_cos', 'is_holiday', 'is_weekend'
]
da_features = ['Wind_Forecast_DA_MW', 'Solar_Forecast_DA_MW',
               'Wind_Forecast_Error_XGB', 'Solar_Forecast_Error_XGB',
               'Residual_Load_DA']
for feat in da_features:
    if feat in data.columns:
        ALL_FEATURES.append(feat)

# Verify all features exist
missing = [f for f in ALL_FEATURES if f not in data.columns]
if missing:
    print(f"\n*** FATAL: Missing features in DataFrame: {missing} ***")
    sys.exit(1)

print(f"  Final dataset: {len(data):,} rows, {len(ALL_FEATURES)} features")
print(f"  Features: {ALL_FEATURES}")

# ================================================================
# 3. TRAIN / TEST SPLIT (80/20 chronological)
# ================================================================
print("\n[3] Train/test split...")

train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data  = data.iloc[train_size:]
y_train = train_data['Price_EUR']
y_test  = test_data['Price_EUR']

print(f"  Train: {len(train_data):,} rows ({train_data.index.min().date()} -> {train_data.index.max().date()})")
print(f"  Test:  {len(test_data):,} rows ({test_data.index.min().date()} -> {test_data.index.max().date()})")

# Verify spike threshold
actual_q90 = y_test.quantile(0.90)
n_spikes = (y_test > SPIKE_THRESHOLD).sum()
print(f"  Spike threshold: {SPIKE_THRESHOLD} EUR (test-set 90th pctl = {actual_q90:.2f})")
print(f"  Spikes in test set: {n_spikes}")

# ================================================================
# 4. FIXED HYPERPARAMETERS (from 03_price_prediction_main.py)
# ================================================================
XGB_PARAMS = dict(
    n_estimators=800, max_depth=5, learning_rate=0.01,
    subsample=0.7, colsample_bytree=0.9,
    reg_lambda=5.0, reg_alpha=1.0,
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
)
SVR_PARAMS = dict(C=100, kernel='rbf', epsilon=0.1, gamma=0.001)

# ================================================================
# 5. DEFINE CONFIGURATIONS
# ================================================================
STAGE1_OUTPUTS = ['Wind_Predicted_MW', 'Solar_Predicted_MW',
                  'Wind_Forecast_Error_XGB', 'Solar_Forecast_Error_XGB']

ALL_WIND_SOLAR = ['Wind_Predicted_MW', 'Solar_Predicted_MW',
                  'Wind_Forecast_DA_MW', 'Solar_Forecast_DA_MW',
                  'Wind_Forecast_Error_XGB', 'Solar_Forecast_Error_XGB',
                  'Residual_Load_DA']

configs = {
    'BASELINE': ALL_FEATURES,
    'NO_STAGE1': [f for f in ALL_FEATURES if f not in STAGE1_OUTPUTS],
    'NO_WIND_AT_ALL': [f for f in ALL_FEATURES if f not in ALL_WIND_SOLAR],
}

print("\n[4] Configurations:")
for name, feats in configs.items():
    removed = [f for f in ALL_FEATURES if f not in feats]
    print(f"  {name:<20s}: {len(feats)} features (removed: {removed})")


# ================================================================
# 6. RUN EACH CONFIGURATION
# ================================================================
def run_config(name, feature_list, train_data, test_data, y_train, y_test):
    """Train XGB+SVR ensemble, re-optimise alpha, evaluate on test set."""
    print(f"\n{'=' * 70}")
    print(f"  CONFIGURATION: {name} ({len(feature_list)} features)")
    print(f"{'=' * 70}")
    t_start = time.time()

    X_train = train_data[feature_list]
    X_test  = test_data[feature_list]

    # Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc  = scaler_X.transform(X_test)
    y_train_sc = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    # XGBoost (with early stopping on last 10% of training)
    xgb_val_size = int(len(X_train) * 0.1)
    model_xgb = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=50)
    model_xgb.fit(
        X_train.iloc[:-xgb_val_size], y_train.iloc[:-xgb_val_size],
        eval_set=[(X_train.iloc[-xgb_val_size:], y_train.iloc[-xgb_val_size:])],
        verbose=False
    )
    pred_xgb = model_xgb.predict(X_test)
    print(f"  XGBoost trained (best_iter={model_xgb.best_iteration})")

    # SVR
    model_svr = SVR(**SVR_PARAMS)
    model_svr.fit(X_train_sc, y_train_sc)
    pred_svr = scaler_y.inverse_transform(
        model_svr.predict(X_test_sc).reshape(-1, 1)).ravel()
    print(f"  SVR trained")

    # Re-optimise alpha on validation set (last 20% of training)
    val_size = int(len(X_train) * 0.2)
    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]
    X_val_sc = scaler_X.transform(X_val)

    pred_xgb_val = model_xgb.predict(X_val)
    pred_svr_val = scaler_y.inverse_transform(
        model_svr.predict(X_val_sc).reshape(-1, 1)).ravel()

    best_alpha, best_val_mae = 0.5, float('inf')
    for a in np.arange(0, 1.01, 0.01):
        blend = a * pred_xgb_val + (1 - a) * pred_svr_val
        mae = mean_absolute_error(y_val, blend)
        if mae < best_val_mae:
            best_alpha, best_val_mae = a, mae

    alpha = round(best_alpha, 2)
    pred_ensemble = alpha * pred_xgb + (1 - alpha) * pred_svr
    print(f"  Alpha re-optimised: {alpha:.2f} (XGB) / {1 - alpha:.2f} (SVR)")

    # Test-set metrics
    r2   = r2_score(y_test, pred_ensemble)
    rmse = np.sqrt(mean_squared_error(y_test, pred_ensemble))
    mae  = mean_absolute_error(y_test, pred_ensemble)

    # Spike detection
    y_true_class = (y_test > SPIKE_THRESHOLD).astype(int)
    y_pred_class = (pred_ensemble > SPIKE_THRESHOLD).astype(int)
    prec = precision_score(y_true_class, y_pred_class, zero_division=0)
    rec  = recall_score(y_true_class, y_pred_class, zero_division=0)
    f1   = f1_score(y_true_class, y_pred_class, zero_division=0)
    cm   = confusion_matrix(y_true_class, y_pred_class, labels=[0, 1])

    elapsed = time.time() - t_start
    print(f"  R2={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}")
    print(f"  Spike P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")
    print(f"  Confusion matrix: TN={cm[0, 0]} FP={cm[0, 1]} FN={cm[1, 0]} TP={cm[1, 1]}")
    print(f"  Elapsed: {elapsed:.1f}s")

    return {
        'Config': name,
        'Features': len(feature_list),
        'Alpha': alpha,
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'Spike_P': prec,
        'Spike_R': rec,
        'Spike_F1': f1,
        'TN': cm[0, 0],
        'FP': cm[0, 1],
        'FN': cm[1, 0],
        'TP': cm[1, 1],
    }


# Run all configurations
results = []
for name, feats in configs.items():
    results.append(run_config(name, feats, train_data, test_data, y_train, y_test))

# ================================================================
# 7. SUMMARY TABLE
# ================================================================
df_results = pd.DataFrame(results)

print("\n\n" + "=" * 90)
print("  GROUPED ABLATION SUMMARY")
print("=" * 90)
print(f"  {'Config':<20s} {'Feat':>4s} {'Alpha':>6s} {'R2':>7s} {'RMSE':>7s} {'MAE':>7s}"
      f" {'Sp_P':>6s} {'Sp_R':>6s} {'Sp_F1':>6s}")
print("  " + "-" * 82)
for _, row in df_results.iterrows():
    print(f"  {row['Config']:<20s} {row['Features']:>4d} {row['Alpha']:>6.2f}"
          f" {row['R2']:>7.4f} {row['RMSE']:>7.2f} {row['MAE']:>7.2f}"
          f" {row['Spike_P']:>6.3f} {row['Spike_R']:>6.3f} {row['Spike_F1']:>6.3f}")

# Delta rows
baseline = df_results.iloc[0]
print("\n  Deltas vs BASELINE:")
print(f"  {'Config':<20s} {'dR2':>8s} {'dRMSE':>8s} {'dMAE':>8s} {'dF1':>8s}")
print("  " + "-" * 48)
for _, row in df_results.iloc[1:].iterrows():
    dr2   = row['R2'] - baseline['R2']
    drmse = row['RMSE'] - baseline['RMSE']
    dmae  = row['MAE'] - baseline['MAE']
    df1   = row['Spike_F1'] - baseline['Spike_F1']
    print(f"  {row['Config']:<20s} {dr2:>+8.4f} {drmse:>+8.2f} {dmae:>+8.2f} {df1:>+8.3f}")

print("\n  Confusion matrices:")
for _, row in df_results.iterrows():
    print(f"  {row['Config']}: TN={row['TN']:,}  FP={row['FP']:,}  FN={row['FN']:,}  TP={row['TP']:,}")

print("=" * 90)

# Save CSV
csv_path = os.path.join(DATA, 'grouped_ablation_summary.csv')
df_results.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")
