#!/usr/bin/env python
# coding: utf-8
"""
conditional_ablation.py
=======================
Tests whether Stage 1 (XGBoost wind/solar predictions) contributes in
conditions where BMRS forecasts are weakest or prices are hardest to predict,
even though it adds no value on average.

Two configurations:
  BASELINE  – all 22 features
  NO_STAGE1 – drop Wind/Solar_Predicted_MW, Wind/Solar_Forecast_Error_XGB

Evaluated on 6 test-set subsets:
  1. All hours
  2. Top quartile of |BMRS wind forecast error| (BMRS got wind most wrong)
  3. Bottom quartile of actual wind generation (low-wind hours)
  4. Top quartile of actual wind generation (high-wind hours)
  5. Price spike hours only (> 130.66 EUR/MWh)
  6. Non-spike hours only (<= 130.66 EUR/MWh)

Same data pipeline, seed, and hyperparameters as grouped_ablation.py.
Ensemble alpha re-optimised per configuration.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             precision_score, recall_score, f1_score)
from sklearn.svm import SVR
import xgboost as xgb
import holidays

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW  = os.path.join(BASE, 'data', 'raw')
DATA = os.path.join(BASE, 'data')

SPIKE_THRESHOLD = 130.66

print("=" * 70)
print("CONDITIONAL ABLATION: Stage 1 value in specific regimes")
print("=" * 70)

# ================================================================
# 1. DATA LOADING (exact replica)
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

# Wind (keep target_mw for actual generation)
df_wind_p['timestamp'] = pd.to_datetime(df_wind_p['timestamp'])
df_wind_p.set_index(df_wind_p['timestamp'].dt.tz_convert(None), inplace=True)
df_wind_hourly = df_wind_p[['target_mw', 'predicted_mw']].resample('h').mean().rename(
    columns={'predicted_mw': 'Wind_Predicted_MW', 'target_mw': 'Wind_Actual_MW'})

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

# Merge (include Wind_Actual_MW as metadata, not as a model feature)
data = df_prices_hourly.join(df_demand_hourly, how='inner')
data = data.join(df_wind_hourly, how='inner')  # brings Wind_Actual_MW + Wind_Predicted_MW
data = data.join(df_solar_hourly, how='inner')
data = data.join(df_gas_hourly, how='inner')
data = data.join(df_co2_hourly, how='inner')

if not df_ws_da_hourly.empty:
    data = data.join(df_ws_da_hourly, how='left')
    for col in df_ws_da_hourly.columns:
        if col in data.columns:
            data[col] = data[col].ffill()

print(f"  Merged: {len(data):,} rows, {data.shape[1]} columns")

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

# Compute BMRS wind forecast error from actual generation (NOT from XGB prediction)
if 'Wind_Forecast_DA_MW' in data.columns and 'Wind_Actual_MW' in data.columns:
    data['BMRS_Wind_Error'] = (data['Wind_Actual_MW'] - data['Wind_Forecast_DA_MW']).abs()
    BMRS_ERROR_SOURCE = "actual wind generation (target_mw) minus BMRS Wind_Forecast_DA_MW"
else:
    # Fallback: use Wind_Forecast_Error_XGB as proxy
    data['BMRS_Wind_Error'] = data['Wind_Forecast_Error_XGB'].abs()
    BMRS_ERROR_SOURCE = "PROXY: |Wind_Forecast_Error_XGB| (= |XGB_pred - BMRS_DA|, not true BMRS error)"

data.dropna(inplace=True)

# Feature list
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

missing = [f for f in ALL_FEATURES if f not in data.columns]
if missing:
    print(f"\n*** FATAL: Missing features: {missing} ***")
    sys.exit(1)

STAGE1_OUTPUTS = ['Wind_Predicted_MW', 'Solar_Predicted_MW',
                  'Wind_Forecast_Error_XGB', 'Solar_Forecast_Error_XGB']
NO_STAGE1_FEATURES = [f for f in ALL_FEATURES if f not in STAGE1_OUTPUTS]

print(f"  Final dataset: {len(data):,} rows, {len(ALL_FEATURES)} features")
print(f"  BMRS error source: {BMRS_ERROR_SOURCE}")

# ================================================================
# 3. TRAIN / TEST SPLIT
# ================================================================
print("\n[3] Train/test split...")

train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data  = data.iloc[train_size:]
y_train = train_data['Price_EUR']
y_test  = test_data['Price_EUR']

print(f"  Train: {len(train_data):,} rows ({train_data.index.min().date()} -> {train_data.index.max().date()})")
print(f"  Test:  {len(test_data):,} rows ({test_data.index.min().date()} -> {test_data.index.max().date()})")

# ================================================================
# 4. HYPERPARAMETERS
# ================================================================
XGB_PARAMS = dict(
    n_estimators=800, max_depth=5, learning_rate=0.01,
    subsample=0.7, colsample_bytree=0.9,
    reg_lambda=5.0, reg_alpha=1.0,
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
)
SVR_PARAMS = dict(C=100, kernel='rbf', epsilon=0.1, gamma=0.001)

# ================================================================
# 5. TRAIN BOTH CONFIGURATIONS
# ================================================================
def train_config(name, feature_list):
    """Train XGB+SVR with re-optimised alpha. Return predictions on full test set."""
    print(f"\n  Training {name} ({len(feature_list)} features)...")
    X_train = train_data[feature_list]
    X_test  = test_data[feature_list]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc  = scaler_X.transform(X_test)
    y_train_sc = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    # XGBoost
    xgb_val_size = int(len(X_train) * 0.1)
    model_xgb = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=50)
    model_xgb.fit(
        X_train.iloc[:-xgb_val_size], y_train.iloc[:-xgb_val_size],
        eval_set=[(X_train.iloc[-xgb_val_size:], y_train.iloc[-xgb_val_size:])],
        verbose=False
    )
    pred_xgb = model_xgb.predict(X_test)

    # SVR
    model_svr = SVR(**SVR_PARAMS)
    model_svr.fit(X_train_sc, y_train_sc)
    pred_svr = scaler_y.inverse_transform(
        model_svr.predict(X_test_sc).reshape(-1, 1)).ravel()

    # Re-optimise alpha
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
    print(f"    Alpha: {alpha:.2f}  best_iter={model_xgb.best_iteration}")

    return pred_ensemble, alpha


print("\n[4] Training models...")
pred_baseline, alpha_baseline = train_config('BASELINE', ALL_FEATURES)
pred_nostage1, alpha_nostage1 = train_config('NO_STAGE1', NO_STAGE1_FEATURES)

# ================================================================
# 6. DEFINE TEST SUBSETS
# ================================================================
print("\n[5] Defining test subsets...")

test_bmrs_error = test_data['BMRS_Wind_Error']
test_wind_actual = test_data['Wind_Actual_MW']
test_prices = y_test

# Quartile thresholds
bmrs_err_q75 = test_bmrs_error.quantile(0.75)
wind_q25 = test_wind_actual.quantile(0.25)
wind_q75 = test_wind_actual.quantile(0.75)

subsets = {
    '1. All hours':            np.ones(len(test_data), dtype=bool),
    '2. BMRS error Q4 (worst)': test_bmrs_error >= bmrs_err_q75,
    '3. Low wind (Q1)':        test_wind_actual <= wind_q25,
    '4. High wind (Q4)':       test_wind_actual >= wind_q75,
    '5. Spike hours (>130.66)': test_prices > SPIKE_THRESHOLD,
    '6. Non-spike hours':      test_prices <= SPIKE_THRESHOLD,
}

print(f"  BMRS error Q75 threshold: {bmrs_err_q75:.1f} MW")
print(f"  Wind actual Q25: {wind_q25:.1f} MW")
print(f"  Wind actual Q75: {wind_q75:.1f} MW")
for name, mask in subsets.items():
    print(f"  {name}: {mask.sum():,} hours")

# ================================================================
# 7. EVALUATE ON EACH SUBSET
# ================================================================
print("\n[6] Evaluating on subsets...")

results = []
for subset_name, mask in subsets.items():
    mask_arr = mask.values if hasattr(mask, 'values') else mask
    n = mask_arr.sum()

    y_true = y_test.values[mask_arr]
    p_base = pred_baseline[mask_arr]
    p_nos1 = pred_nostage1[mask_arr]

    # R² (skip if n < 10 or variance is zero)
    if n >= 10 and np.var(y_true) > 0:
        r2_base = r2_score(y_true, p_base)
        r2_nos1 = r2_score(y_true, p_nos1)
    else:
        r2_base = r2_nos1 = float('nan')

    rmse_base = np.sqrt(mean_squared_error(y_true, p_base))
    rmse_nos1 = np.sqrt(mean_squared_error(y_true, p_nos1))
    mae_base  = mean_absolute_error(y_true, p_base)
    mae_nos1  = mean_absolute_error(y_true, p_nos1)

    # Spike F1 (only meaningful for subsets that contain both classes)
    y_true_class = (y_true > SPIKE_THRESHOLD).astype(int)
    p_base_class = (p_base > SPIKE_THRESHOLD).astype(int)
    p_nos1_class = (p_nos1 > SPIKE_THRESHOLD).astype(int)

    n_pos = y_true_class.sum()
    n_neg = (1 - y_true_class).sum()

    if n_pos > 0 and n_neg > 0:
        f1_base = f1_score(y_true_class, p_base_class, zero_division=0)
        f1_nos1 = f1_score(y_true_class, p_nos1_class, zero_division=0)
    elif n_pos > 0:
        # All spikes — use recall as proxy
        f1_base = recall_score(y_true_class, p_base_class, zero_division=0)
        f1_nos1 = recall_score(y_true_class, p_nos1_class, zero_division=0)
    else:
        # No spikes — F1 not applicable
        f1_base = f1_nos1 = float('nan')

    results.append({
        'Subset': subset_name,
        'N': n,
        'N_spikes': n_pos,
        'Base_R2': r2_base,
        'NoS1_R2': r2_nos1,
        'dR2': r2_nos1 - r2_base if not (np.isnan(r2_base) or np.isnan(r2_nos1)) else float('nan'),
        'Base_RMSE': rmse_base,
        'NoS1_RMSE': rmse_nos1,
        'dRMSE': rmse_nos1 - rmse_base,
        'Base_MAE': mae_base,
        'NoS1_MAE': mae_nos1,
        'dMAE': mae_nos1 - mae_base,
        'Base_F1': f1_base,
        'NoS1_F1': f1_nos1,
        'dF1': f1_nos1 - f1_base if not (np.isnan(f1_base) or np.isnan(f1_nos1)) else float('nan'),
    })

df = pd.DataFrame(results)

# ================================================================
# 8. PRINT RESULTS
# ================================================================
print("\n\n" + "=" * 110)
print("  CONDITIONAL ABLATION: BASELINE vs NO_STAGE1 by test-set subset")
print("=" * 110)
print(f"  BMRS wind forecast error source: {BMRS_ERROR_SOURCE}")
print(f"  Baseline alpha: {alpha_baseline:.2f}   NO_STAGE1 alpha: {alpha_nostage1:.2f}")
print()

# Main comparison table
hdr = (f"  {'Subset':<28s} {'N':>5s} {'Base R2':>8s} {'NoS1 R2':>8s} {'dR2':>8s}"
       f" {'Base RMSE':>10s} {'NoS1 RMSE':>10s} {'dRMSE':>7s}"
       f" {'Base F1':>8s} {'NoS1 F1':>8s} {'dF1':>7s}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for _, row in df.iterrows():
    r2b = f"{row['Base_R2']:.4f}" if not np.isnan(row['Base_R2']) else "   n/a"
    r2n = f"{row['NoS1_R2']:.4f}" if not np.isnan(row['NoS1_R2']) else "   n/a"
    dr2 = f"{row['dR2']:+.4f}" if not np.isnan(row['dR2']) else "   n/a"
    f1b = f"{row['Base_F1']:.3f}" if not np.isnan(row['Base_F1']) else "  n/a"
    f1n = f"{row['NoS1_F1']:.3f}" if not np.isnan(row['NoS1_F1']) else "  n/a"
    df1 = f"{row['dF1']:+.3f}" if not np.isnan(row['dF1']) else "  n/a"
    print(f"  {row['Subset']:<28s} {row['N']:>5d}"
          f" {r2b:>8s} {r2n:>8s} {dr2:>8s}"
          f" {row['Base_RMSE']:>10.2f} {row['NoS1_RMSE']:>10.2f} {row['dRMSE']:>+7.2f}"
          f" {f1b:>8s} {f1n:>8s} {df1:>7s}")

print("\n" + "=" * 110)
print("  INTERPRETATION")
print("=" * 110)
print("  Positive dR2 = NO_STAGE1 is BETTER (Stage 1 hurts)")
print("  Negative dR2 = NO_STAGE1 is WORSE  (Stage 1 helps)")
print("  Positive dRMSE = NO_STAGE1 is WORSE")
print("  Negative dF1 = NO_STAGE1 is WORSE at spike detection")
print()

# Check the key question
key_subsets = ['2. BMRS error Q4 (worst)', '3. Low wind (Q1)', '5. Spike hours (>130.66)']
print("  KEY QUESTION: Does Stage 1 help where BMRS is weakest or during spikes?")
for name in key_subsets:
    row = df[df['Subset'] == name].iloc[0]
    dr2 = row['dR2']
    drmse = row['dRMSE']
    if not np.isnan(dr2):
        if dr2 < -0.005:
            verdict = "YES — Stage 1 helps substantially"
        elif dr2 < -0.001:
            verdict = "MARGINAL — Stage 1 helps slightly"
        elif dr2 > 0.005:
            verdict = "NO — Stage 1 actually hurts"
        else:
            verdict = "NO — negligible difference"
    else:
        verdict = "INCONCLUSIVE"
    print(f"    {name}: dR2={dr2:+.4f}, dRMSE={drmse:+.2f} => {verdict}")

print("=" * 110)
