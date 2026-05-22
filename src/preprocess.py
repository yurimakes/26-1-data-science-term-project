"""
Bank Direct Marketing - Data Preprocessing
- Remove duplicated rows
- Replace 'unknown' with NaN
- Produce 2 imputation versions:
    (A) NaN -> mean (numeric) / mode (categorical, mean-analog)
    (B) NaN -> forward-fill (+ backfill for leading NaN)
- Encode (binary 0/1, one-hot for multi-category)
- Scale numeric columns with StandardScaler
- Save model-ready CSVs
Outputs:
    preprocessed_mean.csv
    preprocessed_ffill.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/raw/bank_direct_marketing_campaigns.csv"
OUT_DIR   = "data/processed"
OUT_MEAN  = os.path.join(OUT_DIR, "preprocessed_mean.csv")
OUT_FFILL = os.path.join(OUT_DIR, "preprocessed_ffill.csv")

BINARY_COLS = ["default", "housing", "loan", "y"]
MULTI_CAT_COLS = ["job", "marital", "education", "contact",
                  "month", "day_of_week", "poutcome"]
NUMERIC_COLS = ["age", "campaign", "pdays", "previous",
                "emp.var.rate", "cons.price.idx", "cons.conf.idx",
                "euribor3m", "nr.employed"]


# ---------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("=" * 70)
print("[1] LOAD")
print("=" * 70)
print(f"Initial shape: {df.shape}")

# ---------------------------------------------------------------
# 2. Remove duplicates
# ---------------------------------------------------------------
dup_cnt = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print(f"\n[2] Removed {dup_cnt} duplicate rows -> shape: {df.shape}")

# ---------------------------------------------------------------
# 3. Replace 'unknown' with NaN
# ---------------------------------------------------------------
obj_cols = df.select_dtypes(include="object").columns.tolist()
unknown_cols = [c for c in obj_cols if (df[c].astype(str) == "unknown").any()]
for c in unknown_cols:
    df[c] = df[c].replace("unknown", np.nan)

print("\n[3] 'unknown' -> NaN")
print(f"Affected columns: {unknown_cols}")
print(f"NaN counts:\n{df[unknown_cols].isna().sum()}")


# ---------------------------------------------------------------
# 4. Imputation strategies
# ---------------------------------------------------------------
def impute_mean(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for c in out.columns:
        if out[c].isna().any():
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].fillna(out[c].mean())
            else:
                out[c] = out[c].fillna(out[c].mode()[0])
    return out


def impute_ffill(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.copy().ffill().bfill()


# ---------------------------------------------------------------
# 5. Encode + Scale
# ---------------------------------------------------------------
def encode_and_scale(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    binary_map = {"yes": 1, "no": 0}
    for col in BINARY_COLS:
        out[col] = out[col].map(binary_map)

    out = pd.get_dummies(out, columns=MULTI_CAT_COLS, drop_first=True)
    bool_cols = out.select_dtypes(include="bool").columns
    out[bool_cols] = out[bool_cols].astype(int)

    scaler = StandardScaler()
    out[NUMERIC_COLS] = scaler.fit_transform(out[NUMERIC_COLS])
    return out


# ---------------------------------------------------------------
# 6. Build two versions and save
# ---------------------------------------------------------------
df_mean  = encode_and_scale(impute_mean(df))
df_ffill = encode_and_scale(impute_ffill(df))

df_mean.to_csv(OUT_MEAN, index=False)
df_ffill.to_csv(OUT_FFILL, index=False)

print("\n" + "=" * 70)
print("[6] SAVED (model-ready)")
print("=" * 70)
print(f"Version A (mean)  -> {OUT_MEAN}")
print(f"  shape: {df_mean.shape}, NaN remaining: {df_mean.isna().sum().sum()}")
print(f"Version B (ffill) -> {OUT_FFILL}")
print(f"  shape: {df_ffill.shape}, NaN remaining: {df_ffill.isna().sum().sum()}")
