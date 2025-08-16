import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

RND = 42  # for reproducible shuffling

# Step 1: Load data
load_df = pd.read_csv("data/Load_history_final.csv")
temp_df = pd.read_csv("data/Temp_history_final.csv")
print(f"Raw load shape: {load_df.shape}")
print(f"Raw temp shape: {temp_df.shape}")

# Step 1.2: Melt to long format
hours = [f"h{i}" for i in range(1,25)]
load_long = load_df.melt(
    id_vars=["zone_id","year","month","day"],
    value_vars=hours, var_name="hour", value_name="load"
)
load_long["hour"] = load_long["hour"].str.extract(r"h(\d+)").astype(int)

temp_long = temp_df.melt(
    id_vars=["station_id","year","month","day"],
    value_vars=hours, var_name="hour", value_name="temp"
)
temp_long["hour"] = temp_long["hour"].str.extract(r"h(\d+)").astype(int)
print(f"load_long shape: {load_long.shape}")
print(f"temp_long shape: {temp_long.shape}")

# Step 1.3: Outlier removal
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

# Step 1.4: Date and dayofweek features
load_long["date"] = pd.to_datetime(load_long[["year","month","day"]])
temp_long["date"] = pd.to_datetime(temp_long[["year","month","day"]])
for df in (load_long, temp_long):
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"]     = df["date"].dt.month
print("Sample load_long with new features:")
print(load_long.head())

# Step 1.5: Shuffle
load_long = load_long.sample(frac=1, random_state=RND).reset_index(drop=True)
temp_long = temp_long.sample(frac=1, random_state=RND).reset_index(drop=True)
print(f"After shuffling → load_long: {load_long.shape}, temp_long: {temp_long.shape}")

# Step 2: Zone–Station Mapping
load_ts = load_long.set_index(["year","month","day","hour","zone_id"])["load"].unstack("zone_id")
temp_ts = temp_long.set_index(["station_id","year","month","day","hour"])["temp"].unstack("station_id")
idx = load_ts.index.intersection(temp_ts.index)
L, T = load_ts.loc[idx], temp_ts.loc[idx]

recs = []
for z in L.columns:
    for s in T.columns:
        r = L[z].corr(T[s])
        if pd.notna(r):
            recs.append({"zone_id": z, "station_id": s, "corr": r})
corr_df = pd.DataFrame(recs)
corr_df["abs_corr"] = corr_df["corr"].abs()
best_map = corr_df.loc[corr_df.groupby("zone_id")["abs_corr"].idxmax()].reset_index(drop=True)

quantiles = best_map["abs_corr"].quantile([0.25, 0.5, 0.75, 0.90])
print("abs_corr quantiles:\n", quantiles, "\n")
thresh = quantiles.loc[0.50]
print(f"Using threshold = 50th percentile = {thresh:.3f}\n")
best_map["strong"] = best_map["abs_corr"] >= thresh
print("Zones flagged WEAK (abs_corr < thresh):")
print(best_map.loc[~best_map["strong"], ["zone_id","station_id","corr","abs_corr"]])

# Step 3: Merge & Split into Train/Val/Test
df_merged = (
    load_long
    .merge(best_map[['zone_id','station_id']], on='zone_id')
    .merge(
        temp_long[['station_id','year','month','day','hour','temp']],
        on=['station_id','year','month','day','hour']
    )
    .rename(columns={'temp':'temperature'})
)

X = df_merged[['zone_id','temperature','hour','dayofweek','month']]
y = df_merged['load']
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, random_state=RND)
X_val, X_test, y_val, y_test    = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=RND)
print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]} rows")

# Step 4: Train GBR
model = Pipeline([
    ("scaler", StandardScaler()),
    ("gbr",    GradientBoostingRegressor(
                   n_estimators=100,
                   learning_rate=0.1,
                   max_depth=3,
                   random_state=RND
               ))
])
start = time.perf_counter()
model.fit(X_train, y_train)
training_time = time.perf_counter() - start
print(f"Training time: {training_time:.2f} seconds")
print("R² train:", model.score(X_train, y_train))
print("R² val:  ", model.score(X_val,   y_val))
print("R² test: ", model.score(X_test,  y_test))

# Step 5: Error analysis for test set
y_pred = model.predict(X_test)
df_meta = (
    df_merged
      .loc[X_test.index, ["zone_id","year","month","day","hour"]]
      .reset_index(drop=True)
      .assign(
         y_true      = y_test,
         y_pred      = y_pred,
         error       = y_test - y_pred,
         abs_error   = np.abs(y_test - y_pred),
         pct_error   = 100 * (y_test - y_pred) / y_test.replace(0, np.nan),
         abs_pct_err = np.abs(100 * (y_test - y_pred) / y_test.replace(0, np.nan))
      )
)
top10_abs = df_meta.nlargest(10, "abs_error")[[
    "zone_id","year","month","day","hour","y_true","y_pred","abs_error","abs_pct_err"
]]
print("Top 10 errors by absolute error:")
print(top10_abs)
top10_pct = df_meta.nlargest(10, "abs_pct_err")[[
    "zone_id","year","month","day","hour","y_true","y_pred","abs_error","pct_error"
]]
print("\nTop 10 errors by percentage error:")
print(top10_pct)

# Step 6: Predict June 1–7, 2008
mask_june = (df_merged["year"] == 2008) & (df_merged["month"] == 6)
X_june_merged = df_merged.loc[mask_june, ["zone_id","temperature","hour","dayofweek","month"]]
y_pred_june   = model.predict(X_june_merged)
out = df_merged.loc[mask_june, ["zone_id","year","month","day","hour"]].reset_index(drop=True)
out["predicted_load"] = y_pred_june
out.to_csv("Load_prediction.csv", index=False)
print("Saved Load_prediction.csv with", len(out), "rows.")
