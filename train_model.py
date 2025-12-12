import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

# ---------------------------------------------------
# 1. Generate Training Data (Simulasi Normal Behavior)
# ---------------------------------------------------

# Fitur:
# temperature, humidity, pir, motion_intensity
normal_data = np.array([
    [
        np.random.uniform(20, 32),        # suhu normal
        np.random.uniform(40, 80),        # humidity normal
        np.random.choice([0, 1], p=[0.7, 0.3]),  # PIR mostly 0
        np.random.uniform(0.0, 0.5)       # low movement intensity
    ]
    for _ in range(800)
])

# ---------------------------------------------------
# 2. Generate Anomaly Data (Gerakan aneh/agresif)
# ---------------------------------------------------
anomaly_data = np.array([
    [
        np.random.uniform(33, 40),        # suhu naik abnormal
        np.random.uniform(20, 30),        # humidity drop
        1,                                 # PIR surely triggered
        np.random.uniform(0.6, 1.0)       # high intensity
    ]
    for _ in range(200)
])

# Gabungkan data
X = np.vstack([normal_data, anomaly_data])

# ---------------------------------------------------
# 3. Train Model (Isolation Forest)
# ---------------------------------------------------
model = IsolationForest(
    n_estimators=200,
    contamination=0.2,
    random_state=42
)
model.fit(X)

# ---------------------------------------------------
# 4. Save Model
# ---------------------------------------------------
joblib.dump(model, "model/model_vigilant.pkl")

print("Model trained & saved as model/model_vigilant.pkl")
