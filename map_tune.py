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