# Power_Load_predictor

##  1. Overview
This project applies **machine learning** to predict hourly electricity demand across **20 zones** for a one-week horizon (**June 1–7, 2008**).  
Two models were implemented and compared:
- **K-Nearest Neighbors (KNN)** – simple, fast, interpretable
- **Gradient Boosting Regressor (GBR)** – complex, more accurate, slower

## 2. Dataset
- **Features:** Zone ID, temperature, hour, day of week, month  
- **Target:** Hourly load (kW) per zone  
- **Split:**  
  - Train: 70%  
  - Validation: 15%  
  - Test: 15%  
- **Source:** Course-provided smart grid datasets  
- **Random State:** `42` for reproducibility

## 3. Methodology

### Data Preprocessing
- Removed outliers  
- Mapped each zone to the most correlated temperature station (Pearson’s r)  
- Classified correlations as **strong** or **weak** using median threshold (0.1635)  
- Reshaped to time-series format for modeling

### Models
a. KNN
KNeighborsRegressor(n_neighbors=5, leaf_size=50)
Pros: Fast, interpretable, no strong assumptions
Cons: Struggles with extreme load spikes & near-zero loads

b.GBR
GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=42)
Pros: Captures non-linear patterns, better accuracy
Cons: Slower training time

## 4. RESULTS:
The KNN model achieved good R² scores on training, validation, and test sets with a training time of about 1.8 seconds, but it struggled with extreme high-load spikes and near-zero loads, leading to larger absolute and percentage errors. The Gradient Boosting Regressor (GBR) delivered higher R² scores and reduced top absolute errors by roughly 45% compared to KNN, although percentage errors for very low loads remained similar, and the training time was notably longer.



