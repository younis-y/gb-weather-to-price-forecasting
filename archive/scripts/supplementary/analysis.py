#!/usr/bin/env python
# coding: utf-8
"""
05_additional_analysis.py
=========================
Task B: Pairwise Wilcoxon Signed-Rank Tests (with Bonferroni correction)
Task C: Leave-One-Out Feature Ablation (all 22 features, XGB+SVR ensemble)
Task D: Battery Arbitrage Numbers (Oracle / AI / Persistence dispatch)

Replicates the exact data pipeline from 03_price_prediction_main.py.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd

from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import xgboost as xgb
import holidays

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW  = os.path.join(BASE, 'data', 'raw')
DATA = os.path.join(BASE, 'data')

print("=" * 70)
print("05_additional_analysis.py")
print("=" * 70)

# ================================================================
# 1. DATA LOADING  (exact replica of 03_price_prediction_main.py)
# ================================================================
print("\n[1] Loading data...")

df_prices  = pd.read_csv(os.path.join(RAW, 'elec_price_hourly_ember.csv'))
df_wind_p  = pd.read_csv(os.path.join(RAW, 'wind_gen_predicted_hourly_xgboost.csv'))
df_solar_p = pd.read_csv(os.path.join(RAW, 'solar_gen_predicted_hourly_xgboost.csv'))
df_demand  = pd.read_csv(os.path.join(RAW, 'elec_demand_outturn_hh_bmrs.csv'))
df_gas     = pd.read_csv(os.path.join(RAW, 'gas_sap_daily_icis.csv'))
df_co2     = pd.read_csv(os.path.join(RAW, 'carbon_ukets_futures_daily_investing.csv'))

print(f"  Prices:    {df_prices.shape}")
print(f"  Wind pred: {df_wind_p.shape}")
print(f"  Solar pred:{df_solar_p.shape}")
print(f"  Demand:    {df_demand.shape}")
print(f"  Gas:       {df_gas.shape}")
print(f"  CO2:       {df_co2.shape}")

# --- DA forecast files (optional, same logic as main script) ---
def strip_tz(idx):
    if hasattr(idx, 'tz') and idx.tz is not None:
        return idx.tz_convert(None)
    return idx

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

HAS_DA_WINDSOLAR = False
path_ws_da = os.path.join(RAW, 'renew_gen_forecast_da_hourly_bmrs.csv')
if os.path.exists(path_ws_da):
    df_ws_da = pd.read_csv(path_ws_da)
    df_ws_da['datetime'] = pd.to_datetime(df_ws_da['datetime'])
    df_ws_da.set_index('datetime', inplace=True)
    df_ws_da.index = strip_tz(df_ws_da.index)
    ws_da_cols = [c for c in df_ws_da.columns if 'Forecast' in c]
    df_ws_da_hourly = df_ws_da[ws_da_cols].resample('h').mean()
    ws_da_days = (df_ws_da_hourly.index.max() - df_ws_da_hourly.index.min()).days
    if ws_da_days >= 30:
        HAS_DA_WINDSOLAR = True
        print(f"  Adding BMRS DA wind/solar forecasts ({ws_da_days} days)")
    else:
        df_ws_da_hourly = pd.DataFrame()
        HAS_DA_WINDSOLAR = False
else:
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

# ================================================================
# 2. FEATURE ENGINEERING  (exact replica)
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

# BMRS DA features (conditional)
if 'Wind_Forecast_DA_MW' in data.columns:
    data['Wind_Forecast_Error_XGB'] = data['Wind_Predicted_MW'] - data['Wind_Forecast_DA_MW']
if 'Solar_Forecast_DA_MW' in data.columns:
    data['Solar_Forecast_Error_XGB'] = data['Solar_Predicted_MW'] - data['Solar_Forecast_DA_MW']
if 'Wind_Forecast_DA_MW' in data.columns and 'Solar_Forecast_DA_MW' in data.columns:
    data['Residual_Load_DA'] = data['Demand_MW'] - (data['Wind_Forecast_DA_MW'] + data['Solar_Forecast_DA_MW'])

data.dropna(inplace=True)

# Build feature list (same order as main script)
feature_cols = [
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
        feature_cols.append(feat)

print(f"  Final dataset: {len(data):,} rows, {len(feature_cols)} features")
print(f"  Features: {feature_cols}")

# ================================================================
# 3. TRAIN / TEST SPLIT & SCALING  (exact replica)
# ================================================================
print("\n[3] Train/test split (80/20 chronological)...")

train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data  = data.iloc[train_size:]

X_train = train_data[feature_cols]
y_train = train_data['Price_EUR']
X_test  = test_data[feature_cols]
y_test  = test_data['Price_EUR']

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_sc = scaler_X.fit_transform(X_train)
X_test_sc  = scaler_X.transform(X_test)
y_train_sc = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

print(f"  Train: {len(train_data):,}  ({train_data.index.min().date()} -> {train_data.index.max().date()})")
print(f"  Test:  {len(test_data):,}   ({test_data.index.min().date()} -> {test_data.index.max().date()})")

# ================================================================
# 4. TRAIN ALL MODELS WITH R^2 > 0.80
#    (exact hyperparameters from 03_price_prediction_main.py)
# ================================================================
print("\n[4] Training models (exact hyperparameters from main script)...")

# --- Fixed hyperparameters as specified ---
XGB_PARAMS = dict(
    n_estimators=800, max_depth=5, learning_rate=0.01,
    subsample=0.7, colsample_bytree=0.9,
    reg_lambda=5.0, reg_alpha=1.0,
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
)
SVR_PARAMS = dict(C=100, kernel='rbf', epsilon=0.1, gamma=0.001)
MLP_PARAMS = dict(
    hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
    alpha=0.001, batch_size=200, learning_rate='adaptive', learning_rate_init=0.001,
    max_iter=500, early_stopping=True, n_iter_no_change=10,
    validation_fraction=0.1, random_state=RANDOM_STATE
)

# ---------- XGBoost ----------
t0 = time.time()
_xgb_val_size = int(len(X_train) * 0.1)
model_xgb = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=50)
model_xgb.fit(
    X_train.iloc[:-_xgb_val_size], y_train.iloc[:-_xgb_val_size],
    eval_set=[(X_train.iloc[-_xgb_val_size:], y_train.iloc[-_xgb_val_size:])],
    verbose=False
)
pred_xgb = model_xgb.predict(X_test)
print(f"  XGBoost     trained ({time.time()-t0:.1f}s)  best_iter={model_xgb.best_iteration}")

# ---------- SVR ----------
t0 = time.time()
model_svr = SVR(**SVR_PARAMS)
model_svr.fit(X_train_sc, y_train_sc)
pred_svr = scaler_y.inverse_transform(model_svr.predict(X_test_sc).reshape(-1, 1)).ravel()
print(f"  SVR         trained ({time.time()-t0:.1f}s)")

# ---------- MLP ----------
t0 = time.time()
model_mlp = MLPRegressor(**MLP_PARAMS)
model_mlp.fit(X_train_sc, y_train)
pred_mlp = model_mlp.predict(X_test_sc)
print(f"  MLP         trained ({time.time()-t0:.1f}s)")

# ---------- XGB+SVR Ensemble (learned alpha on validation MAE) ----------
_val_size = int(len(X_train) * 0.2)
_X_val = X_train.iloc[-_val_size:]
_y_val = y_train.iloc[-_val_size:]
_X_val_sc = scaler_X.transform(_X_val)

_pred_xgb_val = model_xgb.predict(_X_val)
_pred_svr_val = scaler_y.inverse_transform(model_svr.predict(_X_val_sc).reshape(-1, 1)).ravel()

best_alpha, best_val_mae = 0.5, float('inf')
for _a in np.arange(0, 1.01, 0.01):
    _blend = _a * _pred_xgb_val + (1 - _a) * _pred_svr_val
    _mae = mean_absolute_error(_y_val, _blend)
    if _mae < best_val_mae:
        best_alpha, best_val_mae = _a, _mae

ENSEMBLE_ALPHA = round(best_alpha, 2)
pred_ensemble = ENSEMBLE_ALPHA * pred_xgb + (1 - ENSEMBLE_ALPHA) * pred_svr
print(f"  Ensemble    alpha={ENSEMBLE_ALPHA:.2f} (XGB) / {1-ENSEMBLE_ALPHA:.2f} (SVR)")

# Persistence baseline (same as main script)
pred_persist = test_data['Price_Lag_24'].values

# Collect all predictions
all_models = {
    'XGB+SVR Ensemble': pred_ensemble,
    'XGBoost':          pred_xgb,
    'SVR':              pred_svr,
    'MLP':              pred_mlp,
}

print("\n  Model Performance on Test Set:")
print(f"  {'Model':<22s}  {'R2':>8s}  {'RMSE':>8s}  {'MAE':>8s}")
print("  " + "-" * 50)
for name, preds in all_models.items():
    r2   = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    print(f"  {name:<22s}  {r2:>8.4f}  {rmse:>8.2f}  {mae:>8.2f}")

# Filter to R^2 > 0.80 only
models_above_80 = {k: v for k, v in all_models.items()
                   if r2_score(y_test, v) > 0.80}
print(f"\n  Models with R2 > 0.80: {list(models_above_80.keys())}")

# ================================================================
# TASK B: PAIRWISE WILCOXON SIGNED-RANK TESTS
# ================================================================
print("\n" + "=" * 70)
print("TASK B: PAIRWISE WILCOXON SIGNED-RANK TESTS")
print("=" * 70)

model_names = list(models_above_80.keys())
n_models = len(model_names)
n_comparisons = n_models * (n_models - 1) // 2

# Compute squared errors for each model
squared_errors = {}
for name, preds in models_above_80.items():
    squared_errors[name] = (y_test.values - preds) ** 2

# Pairwise tests
wilcoxon_rows = []
print(f"\n  Number of models: {n_models}")
print(f"  Number of pairwise comparisons: {n_comparisons}")
print(f"  Bonferroni correction factor: {n_comparisons}")
print()

for i in range(n_models):
    for j in range(i + 1, n_models):
        m1, m2 = model_names[i], model_names[j]
        se1, se2 = squared_errors[m1], squared_errors[m2]
        stat, p_raw = wilcoxon(se1, se2)
        p_corr = min(p_raw * n_comparisons, 1.0)
        sig = '***' if p_corr < 0.001 else '**' if p_corr < 0.01 else '*' if p_corr < 0.05 else 'ns'
        wilcoxon_rows.append({
            'Model_A': m1, 'Model_B': m2,
            'W_statistic': stat,
            'p_raw': p_raw,
            'p_corrected': p_corr,
            'significance': sig
        })

df_wilcoxon = pd.DataFrame(wilcoxon_rows)

# Print triangular matrix
print("  Triangular matrix of Bonferroni-corrected p-values:")
print()

# Build matrix
p_matrix = pd.DataFrame(np.nan, index=model_names, columns=model_names)
for _, row in df_wilcoxon.iterrows():
    p_matrix.loc[row['Model_A'], row['Model_B']] = row['p_corrected']

# Print header
max_name_len = max(len(n) for n in model_names)
header = " " * (max_name_len + 2)
for name in model_names[1:]:
    header += f"{name:>22s}"
print(header)

for i, m1 in enumerate(model_names[:-1]):
    line = f"  {m1:<{max_name_len}s}"
    for j, m2 in enumerate(model_names[1:], 1):
        if j > i:
            val = p_matrix.loc[m1, m2]
            if np.isnan(val):
                line += f"{'---':>22s}"
            else:
                sig = '***' if val < 0.001 else '**' if val < 0.01 else '*' if val < 0.05 else 'ns'
                line += f"  {val:.6f} {sig:>4s}"
        else:
            line += " " * 22
    print(line)

print()
print("  Full results table:")
print(df_wilcoxon.to_string(index=False))

# Save
wilcoxon_path = os.path.join(DATA, 'wilcoxon_pairwise.csv')
df_wilcoxon.to_csv(wilcoxon_path, index=False)
print(f"\n  Saved: {wilcoxon_path}")

# ================================================================
# TASK C: LEAVE-ONE-OUT FEATURE ABLATION (all features)
# ================================================================
print("\n" + "=" * 70)
print("TASK C: LEAVE-ONE-OUT FEATURE ABLATION (XGB+SVR Ensemble)")
print("=" * 70)

# Baseline ensemble R^2
baseline_r2 = r2_score(y_test, pred_ensemble)
print(f"  Baseline ensemble R2: {baseline_r2:.6f}")
print(f"  Ensemble alpha: {ENSEMBLE_ALPHA:.2f}")
print(f"  Total features to ablate: {len(feature_cols)}")
print()

ablation_results = []

for feat_idx, feat_name in enumerate(feature_cols):
    t0 = time.time()

    # Remove this feature from train and test
    cols_ablated = [c for c in feature_cols if c != feat_name]

    X_train_abl = train_data[cols_ablated]
    X_test_abl  = test_data[cols_ablated]

    # Re-fit scaler on ablated features
    scaler_X_abl = StandardScaler()
    scaler_y_abl = StandardScaler()
    X_train_abl_sc = scaler_X_abl.fit_transform(X_train_abl)
    X_test_abl_sc  = scaler_X_abl.transform(X_test_abl)
    y_train_abl_sc = scaler_y_abl.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    # Retrain XGBoost (with early stopping)
    xgb_abl = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=50)
    xgb_abl.fit(
        X_train_abl.iloc[:-_xgb_val_size], y_train.iloc[:-_xgb_val_size],
        eval_set=[(X_train_abl.iloc[-_xgb_val_size:], y_train.iloc[-_xgb_val_size:])],
        verbose=False
    )
    pred_xgb_abl = xgb_abl.predict(X_test_abl)

    # Retrain SVR
    svr_abl = SVR(**SVR_PARAMS)
    svr_abl.fit(X_train_abl_sc, y_train_abl_sc)
    pred_svr_abl = scaler_y_abl.inverse_transform(
        svr_abl.predict(X_test_abl_sc).reshape(-1, 1)).ravel()

    # Ensemble with the SAME alpha
    pred_ens_abl = ENSEMBLE_ALPHA * pred_xgb_abl + (1 - ENSEMBLE_ALPHA) * pred_svr_abl
    abl_r2 = r2_score(y_test, pred_ens_abl)
    delta_r2 = baseline_r2 - abl_r2

    elapsed = time.time() - t0
    ablation_results.append({
        'Feature': feat_name,
        'Baseline_R2': baseline_r2,
        'Ablated_R2': abl_r2,
        'Delta_R2': delta_r2,
    })
    print(f"  [{feat_idx+1:2d}/{len(feature_cols)}] Remove {feat_name:<30s}  "
          f"R2={abl_r2:.6f}  Delta_R2={delta_r2:+.6f}  ({elapsed:.1f}s)")

# Sort descending by Delta_R2
df_ablation = pd.DataFrame(ablation_results).sort_values('Delta_R2', ascending=False).reset_index(drop=True)

print("\n" + "-" * 70)
print("  FULL ABLATION RESULTS (sorted by Delta_R2 descending):")
print("-" * 70)
print(f"  {'Rank':>4s}  {'Feature':<32s}  {'Ablated_R2':>11s}  {'Delta_R2':>10s}")
print("  " + "-" * 62)
for rank, (_, row) in enumerate(df_ablation.iterrows(), 1):
    print(f"  {rank:>4d}  {row['Feature']:<32s}  {row['Ablated_R2']:>11.6f}  {row['Delta_R2']:>+10.6f}")

# Comparison: Wind_Predicted_MW vs sum of lag features
wind_delta = df_ablation.loc[df_ablation['Feature'] == 'Wind_Predicted_MW', 'Delta_R2'].values[0]
lag_features = ['Price_Lag_24', 'Price_Lag_168', 'Price_Roll_Mean_24']
lag_sum = df_ablation.loc[df_ablation['Feature'].isin(lag_features), 'Delta_R2'].sum()

print(f"\n  Key comparison:")
print(f"    Wind_Predicted_MW Delta_R2:                         {wind_delta:+.6f}")
print(f"    Sum(Price_Lag_24 + Price_Lag_168 + Price_Roll_Mean_24): {lag_sum:+.6f}")
if wind_delta > lag_sum:
    print(f"    --> Wind_Predicted_MW alone is MORE important than the three lag features combined")
else:
    print(f"    --> The three lag features combined are MORE important than Wind_Predicted_MW alone")
ratio = wind_delta / lag_sum if lag_sum != 0 else float('inf')
print(f"    Ratio (Wind / Lag_sum): {ratio:.2f}x")

# Save
ablation_path = os.path.join(DATA, 'feature_ablation_full.csv')
df_ablation.to_csv(ablation_path, index=False)
print(f"\n  Saved: {ablation_path}")

# ================================================================
# TASK D: BATTERY ARBITRAGE NUMBERS
# ================================================================
print("\n" + "=" * 70)
print("TASK D: BATTERY ARBITRAGE SIMULATION")
print("=" * 70)

import pulp

BATTERY_CAPACITY_MWH = 200
BATTERY_POWER_MW     = 100
EFFICIENCY_RTE       = 0.90
DEGRADATION_COST     = 0.1  # EUR/MWh

print(f"  System: {BATTERY_POWER_MW} MW / {BATTERY_CAPACITY_MWH} MWh")
print(f"  RTE: {EFFICIENCY_RTE*100:.0f}%")
print(f"  Degradation: {DEGRADATION_COST} EUR/MWh")

# --- LP optimizer (exact replica from main script) ---
def optimize_daily_operation(prices_signal, prices_settlement, date_label=""):
    T = len(prices_signal)
    if T < 24:
        return 0.0, 0.0

    prob = pulp.LpProblem(f"Battery_Opt_{date_label}", pulp.LpMaximize)
    c = pulp.LpVariable.dicts("Charge",    range(T), 0, BATTERY_POWER_MW)
    d = pulp.LpVariable.dicts("Discharge", range(T), 0, BATTERY_POWER_MW)
    s = pulp.LpVariable.dicts("SoC",       range(T), 0, BATTERY_CAPACITY_MWH)
    z = pulp.LpVariable.dicts("Mode",      range(T), cat='Binary')

    eta_one_way = np.sqrt(EFFICIENCY_RTE)

    prob += pulp.lpSum([
        (d[t] * prices_signal[t]) - (c[t] * prices_signal[t]) - (DEGRADATION_COST * (c[t] + d[t]))
        for t in range(T)
    ])

    for t in range(T):
        prob += c[t] <= BATTERY_POWER_MW * z[t]
        prob += d[t] <= BATTERY_POWER_MW * (1 - z[t])
        if t == 0:
            prob += s[t] == 0 + (c[t] * eta_one_way) - (d[t] / eta_one_way)
        else:
            prob += s[t] == s[t-1] + (c[t] * eta_one_way) - (d[t] / eta_one_way)

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    charge_schedule    = np.array([pulp.value(c[t]) for t in range(T)])
    discharge_schedule = np.array([pulp.value(d[t]) for t in range(T)])

    real_revenue   = np.sum(discharge_schedule * prices_settlement)
    real_cost      = np.sum(charge_schedule * prices_settlement)
    degradation    = DEGRADATION_COST * np.sum(charge_schedule + discharge_schedule)
    real_profit    = real_revenue - real_cost - degradation
    total_energy   = np.sum(discharge_schedule)

    return real_profit, total_energy

# --- Build simulation dataframe ---
df_sim = pd.DataFrame({
    'Actual_Price':        y_test.values,
    'Ensemble_Prediction': pred_ensemble,
}, index=test_data.index)

# Persistence signal = yesterday's actual prices (Price_Lag_24)
df_sim['Persistence_Price'] = test_data['Price_Lag_24'].values

unique_days = df_sim.index.normalize().unique()
n_days = len(unique_days)
print(f"  Simulation period: {n_days} days")

results_oracle = []
results_ai     = []
results_persist = []

for i, day in enumerate(unique_days):
    day_data = df_sim[df_sim.index.normalize() == day]
    if len(day_data) < 24:
        continue
    day_str = day.strftime('%Y-%m-%d')
    prices_actual = day_data['Actual_Price'].values
    prices_pred   = day_data['Ensemble_Prediction'].values
    prices_lag24  = day_data['Persistence_Price'].values

    # Oracle: optimise on actual, settle on actual
    profit_oracle, vol_oracle = optimize_daily_operation(prices_actual, prices_actual, f"Oracle_{day_str}")
    results_oracle.append({'Date': day, 'Profit': profit_oracle, 'Volume_MWh': vol_oracle})

    # AI forecast dispatch
    profit_ai, vol_ai = optimize_daily_operation(prices_pred, prices_actual, f"AI_{day_str}")
    results_ai.append({'Date': day, 'Profit': profit_ai, 'Volume_MWh': vol_ai})

    # Persistence dispatch: use yesterday's prices as signal
    profit_pers, vol_pers = optimize_daily_operation(prices_lag24, prices_actual, f"Pers_{day_str}")
    results_persist.append({'Date': day, 'Profit': profit_pers, 'Volume_MWh': vol_pers})

    if i % 50 == 0:
        print(f"    Progress: {i}/{n_days} days...")

print(f"    Progress: {n_days}/{n_days} days... done")

df_oracle  = pd.DataFrame(results_oracle).set_index('Date')
df_ai      = pd.DataFrame(results_ai).set_index('Date')
df_persist_bat = pd.DataFrame(results_persist).set_index('Date')

total_oracle  = df_oracle['Profit'].sum()
total_ai      = df_ai['Profit'].sum()
total_persist = df_persist_bat['Profit'].sum()

profit_gap = (total_oracle - total_ai) / total_oracle * 100 if total_oracle != 0 else 0
persist_gap = (total_oracle - total_persist) / total_oracle * 100 if total_oracle != 0 else 0

print(f"\n  {'Strategy':<25s}  {'Cumulative Profit':>18s}  {'Gap vs Oracle':>14s}")
print("  " + "-" * 60)
print(f"  {'Oracle (perfect info)':<25s}  EUR {total_oracle:>14,.2f}  {'---':>14s}")
print(f"  {'AI Forecast (ensemble)':<25s}  EUR {total_ai:>14,.2f}  {profit_gap:>13.1f}%")
print(f"  {'Persistence (24h lag)':<25s}  EUR {total_persist:>14,.2f}  {persist_gap:>13.1f}%")

ai_capture     = (total_ai / total_oracle * 100) if total_oracle != 0 else 0
persist_capture = (total_persist / total_oracle * 100) if total_oracle != 0 else 0
print(f"\n  Revenue capture:  AI = {ai_capture:.1f}%,  Persistence = {persist_capture:.1f}%")

# --- Bootstrap 95% CIs on daily profit differences (10000 resamples) ---
print("\n  Bootstrap 95% CIs on daily profit differences (10,000 resamples):")

daily_profit_oracle = df_oracle['Profit'].values
daily_profit_ai     = df_ai['Profit'].values
daily_profit_pers   = df_persist_bat['Profit'].values

# Align lengths (in case any days were skipped)
n_sim_days = min(len(daily_profit_oracle), len(daily_profit_ai), len(daily_profit_pers))
daily_profit_oracle = daily_profit_oracle[:n_sim_days]
daily_profit_ai     = daily_profit_ai[:n_sim_days]
daily_profit_pers   = daily_profit_pers[:n_sim_days]

diff_oracle_ai   = daily_profit_oracle - daily_profit_ai
diff_oracle_pers  = daily_profit_oracle - daily_profit_pers
diff_ai_pers      = daily_profit_ai - daily_profit_pers

n_boot = 10000

def bootstrap_mean_ci(diffs, n_boot=10000):
    boot_means = np.empty(n_boot)
    n = len(diffs)
    for b in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        boot_means[b] = np.mean(diffs[idx])
    ci_lo = np.percentile(boot_means, 2.5)
    ci_hi = np.percentile(boot_means, 97.5)
    return np.mean(diffs), ci_lo, ci_hi

for label, diffs in [("Oracle - AI",       diff_oracle_ai),
                      ("Oracle - Persistence", diff_oracle_pers),
                      ("AI - Persistence",     diff_ai_pers)]:
    mean_diff, ci_lo, ci_hi = bootstrap_mean_ci(diffs, n_boot)
    print(f"    {label:<25s}: mean = EUR {mean_diff:>9.2f}/day  "
          f"95% CI [{ci_lo:>9.2f}, {ci_hi:>9.2f}]")

# Save battery results
battery_rows = [
    {'Strategy': 'Oracle', 'Cumulative_Profit_EUR': total_oracle,
     'Gap_vs_Oracle_pct': 0.0, 'Revenue_Capture_pct': 100.0},
    {'Strategy': 'AI_Forecast', 'Cumulative_Profit_EUR': total_ai,
     'Gap_vs_Oracle_pct': profit_gap, 'Revenue_Capture_pct': ai_capture},
    {'Strategy': 'Persistence', 'Cumulative_Profit_EUR': total_persist,
     'Gap_vs_Oracle_pct': persist_gap, 'Revenue_Capture_pct': persist_capture},
]
df_battery = pd.DataFrame(battery_rows)

battery_path = os.path.join(DATA, 'battery_results.csv')
df_battery.to_csv(battery_path, index=False)
print(f"\n  Saved: {battery_path}")

# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "=" * 70)
print("ALL RESULTS SUMMARY")
print("=" * 70)

print("\n--- Task B: Wilcoxon Pairwise Tests ---")
print(df_wilcoxon.to_string(index=False))

print("\n--- Task C: Feature Ablation (top 10) ---")
print(df_ablation.head(10).to_string(index=False))
print(f"\n  Wind_Predicted_MW Delta_R2:     {wind_delta:+.6f}")
print(f"  Sum(3 lag features) Delta_R2:   {lag_sum:+.6f}")

print("\n--- Task D: Battery Arbitrage ---")
print(df_battery.to_string(index=False))

print(f"\n  Oracle profit:      EUR {total_oracle:>14,.2f}")
print(f"  AI profit:          EUR {total_ai:>14,.2f}")
print(f"  Persistence profit: EUR {total_persist:>14,.2f}")
print(f"  Profit gap (Oracle-AI)/Oracle: {profit_gap:.1f}%")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
