import pandas as pd
import numpy as np

RND = 42  # for reproducible shuffling
load_df = pd.read_csv("data/Load_history_final.csv")
temp_df = pd.read_csv("data/Temp_history_final.csv")
print(f"Raw load shape: {load_df.shape}")
print(f"Raw temp shape: {temp_df.shape}")
hours = [f"h{i}" for i in range(1,25)]
load_long = load_df.melt(
    id_vars=["zone_id","year","month","day"],
    value_vars=hours,
    var_name="hour",
    value_name="load"
)
load_long["hour"] = load_long["hour"].str.extract(r"h(\d+)").astype(int)
temp_long = temp_df.melt(
    id_vars=["station_id","year","month","day"],
    value_vars=hours,
    var_name="hour",
    value_name="temp"
)
temp_long["hour"] = temp_long["hour"].str.extract(r"h(\d+)").astype(int)
print(f"load_long shape: {load_long.shape}")
print(f"temp_long shape: {temp_long.shape}")
def remove_outliers(df, group_col, value_col):
    Q1 = df.groupby(group_col)[value_col].transform(lambda x: x.quantile(0.25))
    Q3 = df.groupby(group_col)[value_col].transform(lambda x: x.quantile(0.75))
    IQR = Q3 - Q1
    return df.loc[
        (df[value_col] >= Q1 - 1.5 * IQR) &
        (df[value_col] <= Q3 + 1.5 * IQR)
    ].reset_index(drop=True)
load_long = remove_outliers(load_long, "zone_id", "load")
temp_long = remove_outliers(temp_long, "station_id", "temp")
print(f"After outlier removal → load_long: {load_long.shape}, temp_long: {temp_long.shape}")
load_long["date"] = pd.to_datetime(load_long[["year","month","day"]])
temp_long["date"] = pd.to_datetime(temp_long[["year","month","day"]])

for df in (load_long, temp_long):
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"]     = df["date"].dt.month

print("Sample load_long with new features:")
print(load_long.head())
load_long = load_long.sample(frac=1, random_state=RND).reset_index(drop=True)
temp_long = temp_long.sample(frac=1, random_state=RND).reset_index(drop=True)
print(f"After shuffling → load_long: {load_long.shape}, temp_long: {temp_long.shape}")

# Step 2: Zone–Station Mapping (with strong/weak flag)

import pandas as pd

# 2.1 Pivot to time-series
load_ts = load_long.set_index(
    ["year","month","day","hour","zone_id"]
)["load"].unstack("zone_id")
temp_ts = temp_long.set_index(
    ["station_id","year","month","day","hour"]
)["temp"].unstack("station_id")

# 2.2 Align timestamps
idx = load_ts.index.intersection(temp_ts.index)
L, T = load_ts.loc[idx], temp_ts.loc[idx]

# 2.3 Compute correlations
recs = []
for z in L.columns:
    for s in T.columns:
        r = L[z].corr(T[s])
        if pd.notna(r):
            recs.append({"zone_id": z, "station_id": s, "corr": r})
corr_df = pd.DataFrame(recs)
corr_df["abs_corr"] = corr_df["corr"].abs()

# 2.4 Pick best station per zone
best_map = (
    corr_df
    .loc[corr_df.groupby("zone_id")["abs_corr"].idxmax()]
    .reset_index(drop=True)
)
quantiles = best_map["abs_corr"].quantile([0.25, 0.5, 0.75, 0.90])
print("abs_corr quantiles:\n", quantiles, "\n")

# 3) Choose a new threshold—e.g., the 75th percentile of best abs_corr
thresh = quantiles.loc[0.50]
print(f"Using threshold = 50th percentile = {thresh:.3f}\n")

# 4) Flag strong vs. weak using this adaptive threshold
best_map["strong"] = best_map["abs_corr"] >= thresh

# 5) Show results
print("New zone→station mapping with adaptive threshold:\n", best_map, "\n")
print("Zones flagged WEAK (abs_corr < thresh):")
print(best_map.loc[~best_map["strong"], ["zone_id","station_id","corr","abs_corr"]])


# Step 3: Merge & Split into Train / Validation / Test

from sklearn.model_selection import train_test_split

# 3.1 Merge load_long + best_map + temp_long 
df_merged = (
    load_long
    .merge(best_map[['zone_id','station_id']], on='zone_id')
    .merge(
        temp_long[['station_id','year','month','day','hour','temp']],
        on=['station_id','year','month','day','hour']
    )
    .rename(columns={'temp':'temperature'})
)

# 3.2 Define features and target
X = df_merged[['zone_id','temperature','hour','dayofweek','month']]
y = df_merged['load']

# 3.3 First carve out 30% for val+test
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42
)

# 3.4 Split that 30% into 15% val and 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp,
    test_size=0.50,
    random_state=42
)

# 3.5 Sanity check
print(f"Train:      {X_train.shape[0]} rows")
print(f"Validation: {X_val.shape[0]} rows")
print(f"Test:       {X_test.shape[0]} rows")
print(f"Total:      {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]} rows")


import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn",    KNeighborsRegressor(n_neighbors=5, leaf_size=50))
])

start = time.perf_counter()
model.fit(X_train, y_train)
training_time = time.perf_counter() - start

print(f"Training time: {training_time:.2f} seconds")
print("R² train:", model.score(X_train, y_train))
print("R² val:  ", model.score(X_val,   y_val))
print("R² test: ", model.score(X_test,  y_test))
import pandas as pd
import numpy as np


# 1) Get test‐set predictions 
y_pred_test = model.predict(X_test)
y_true_test = y_test.values

# 2) Build a DataFrame with metadata and errors
df_test = df_merged.loc[X_test.index, ["zone_id","year","month","day","hour"]].copy()
df_test = df_test.reset_index(drop=True)
df_test["y_true"]      = y_true_test
df_test["y_pred"]      = y_pred_test
df_test["error"]       = df_test["y_true"] - df_test["y_pred"]
df_test["abs_error"]   = df_test["error"].abs()
df_test["pct_error"]   = 100 * df_test["error"] / df_test["y_true"].replace(0, np.nan)
df_test["abs_pct_err"] = df_test["pct_error"].abs()

# 3) Select the top 10 by absolute error (or abs_pct_err)
top10_abs = df_test.nlargest(10, "abs_error")
print("Top 10 by absolute error:")
print(top10_abs[[
    "zone_id","year","month","day","hour",
    "y_true","y_pred","abs_error","abs_pct_err"
]])

# top 10 by percentage error, excluding zero‐true cases:
top10_pct = df_test.dropna(subset=["pct_error"]).nlargest(10, "abs_pct_err")
print("\nTop 10 by percentage error:")
print(top10_pct[[
    "zone_id","year","month","day","hour",
    "y_true","y_pred","abs_error","pct_error"
]])
