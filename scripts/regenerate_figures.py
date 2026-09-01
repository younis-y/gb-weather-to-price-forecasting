"""
Regenerate figures/dl_model_comparison.png and figures/price_model_results.png
with report-aligned numbers (Table 10/11).

Run from scripts/:
    cd scripts && python regenerate_figures.py

To change numbers, edit MODEL_RESULTS and DL_RESULTS below.
For the scatter / error / time-series panels (Figure 7, panels B/C/D), the
script loads ensemble predictions from cache. Adjust the paths in the
PREDICTIONS section if your cache layout differs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. REPORT NUMBERS  --  edit here
# =============================================================================
# Table 10 / 11 -- full ranking
MODEL_RESULTS = [
    {"Model": "XGB+SVR Ensemble", "Category": "Classical ML",         "R2": 0.8597, "RMSE": 13.46, "MAE":  9.44},
    {"Model": "XGBoost",          "Category": "Classical ML",         "R2": 0.8491, "RMSE": 13.97, "MAE":  9.63},
    {"Model": "TCN",              "Category": "Deep Learning",        "R2": 0.8452, "RMSE": 14.15, "MAE": 10.21},
    {"Model": "SVR",              "Category": "Classical ML",         "R2": 0.8247, "RMSE": 15.05, "MAE": 10.75},
    {"Model": "DNN",              "Category": "Deep Learning",        "R2": 0.8242, "RMSE": 15.07, "MAE": 10.78},
    {"Model": "LSTM",             "Category": "Deep Learning",        "R2": 0.7497, "RMSE": 17.99, "MAE": 12.91},
    {"Model": "24h Persistence",  "Category": "Statistical Baseline", "R2": 0.3328, "RMSE": 29.37, "MAE": 18.91},
    {"Model": "ARIMA(5,1,0)",     "Category": "Statistical Baseline", "R2": -3.12,  "RMSE": 72.94, "MAE": 64.31},
]

# Figure 9 -- DL only
DL_RESULTS = [
    {"Model": "TCN",  "R2": 0.8452, "RMSE": 14.15, "MAE": 10.21},
    {"Model": "DNN",  "R2": 0.8242, "RMSE": 15.07, "MAE": 10.78},
    {"Model": "LSTM", "R2": 0.7497, "RMSE": 17.99, "MAE": 12.91},
]

# =============================================================================
# 2. PATHS  --  edit if your layout differs
# =============================================================================
CACHE_DIR = Path("../cache")
FIG_DIR   = Path("../figures")
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Used for Figure 7 panels B (scatter), C (error hist), D (time series).
# Set any to None and the script will skip that panel and leave it blank.
Y_TEST_FILE        = CACHE_DIR / "y_test.npy"
PRED_ENSEMBLE_FILE = CACHE_DIR / "pred_ensemble.npy"
TEST_INDEX_FILE    = CACHE_DIR / "test_index.npy"   # datetime64 index, optional

# =============================================================================
# 3. STYLE
# =============================================================================
sns.set_style("whitegrid")
CAT_PALETTE = {
    "Classical ML":         "#2980b9",
    "Deep Learning":        "#e74c3c",
    "Statistical Baseline": "#95a5a6",
}
DL_BAR_PALETTE = ["#2ecc71", "#3498db", "#e67e22"]


# =============================================================================
# Figure 9 -- DL model comparison
# =============================================================================
def make_dl_comparison():
    dl_df = pd.DataFrame(DL_RESULTS).sort_values("R2", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pal = DL_BAR_PALETTE[: len(dl_df)]

    sns.barplot(x="R2",   y="Model", data=dl_df, palette=pal, ax=axes[0])
    sns.barplot(x="RMSE", y="Model", data=dl_df, palette=pal, ax=axes[1])
    sns.barplot(x="MAE",  y="Model", data=dl_df, palette=pal, ax=axes[2])

    plt.tight_layout()
    out = FIG_DIR / "dl_model_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


# =============================================================================
# Figure 7 -- Price model results (4-panel)
# =============================================================================
def _try_load(path):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f"  [warn] {p} not found -- panel will be blank")
        return None
    arr = np.load(p, allow_pickle=True)
    return arr


def make_price_results():
    ranking = (
        pd.DataFrame(MODEL_RESULTS)
        .sort_values("R2", ascending=False)
        .reset_index(drop=True)
    )

    y_test         = _try_load(Y_TEST_FILE)
    pred_ensemble  = _try_load(PRED_ENSEMBLE_FILE)
    test_index_raw = _try_load(TEST_INDEX_FILE)
    test_index     = pd.to_datetime(test_index_raw) if test_index_raw is not None else None

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # A. Model Ranking ------------------------------------------------------
    sns.barplot(
        x="R2", y="Model", hue="Category",
        data=ranking, palette=CAT_PALETTE,
        dodge=False, ax=axes[0, 0],
    )
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].legend(title="Category", loc="lower right", fontsize=9)

    # B. Goodness-of-fit scatter -------------------------------------------
    if y_test is not None and pred_ensemble is not None:
        axes[0, 1].scatter(y_test, pred_ensemble, alpha=0.3, color="purple", s=10)
        lo, hi = float(np.min(y_test)), float(np.max(y_test))
        axes[0, 1].plot([lo, hi], [lo, hi], "r--", lw=2)
    axes[0, 1].set_xlabel("Actual Price (EUR)")
    axes[0, 1].set_ylabel("Predicted Price (EUR)")

    # C. Error distribution -------------------------------------------------
    if y_test is not None and pred_ensemble is not None:
        errors = y_test - pred_ensemble
        sns.histplot(errors, bins=50, kde=True, color="teal", ax=axes[1, 0])
        axes[1, 0].axvline(x=0, color="red", linestyle="--", lw=2)
    axes[1, 0].set_xlabel("Error (EUR)")

    # D. 2-week time series sample -----------------------------------------
    window = 336
    if y_test is not None and pred_ensemble is not None:
        x = test_index[:window] if test_index is not None else np.arange(window)
        axes[1, 1].plot(x, y_test[:window],
                        label="Actual Price", color="black", alpha=0.5)
        axes[1, 1].plot(x, pred_ensemble[:window],
                        label="Ensemble Forecast", color="purple", linestyle="--")
        axes[1, 1].legend()
        if test_index is not None:
            plt.setp(axes[1, 1].get_xticklabels(), rotation=45)
    axes[1, 1].set_ylabel("Price (EUR/MWh)")

    plt.tight_layout()
    out = FIG_DIR / "price_model_results.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    print("Regenerating figures with report-aligned numbers...")
    make_dl_comparison()
    make_price_results()
    print("Done.")
