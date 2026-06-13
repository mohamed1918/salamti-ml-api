import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

# =========================
# 1. Generate synthetic data
# =========================
n = 5000

peakG = np.abs(np.random.normal(3, 2, n))
gyroX = np.random.normal(0, 120, n)
gyroY = np.random.normal(0, 120, n)

gyro_mag = np.sqrt(gyroX**2 + gyroY**2)

# =========================
# 2. Labeling rule (ground truth)
# =========================
g_score = np.clip(peakG / 10, 0, 1)
gyro_score = np.clip(gyro_mag / 250, 0, 1)

S = 0.6 * g_score + 0.4 * gyro_score

y = []
for s in S:
    if s >= 0.75:
        y.append("HIGH")
    elif s >= 0.45:
        y.append("MEDIUM")
    elif s > 0.2:
        y.append("LOW")
    else:
        y.append("UNKNOWN")

# =========================
# 3. Training data
# =========================
X = np.column_stack((peakG, gyro_mag))

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# =========================
# 4. Save model
# =========================
joblib.dump(model, "model.pkl")

print("Model trained and saved successfully!")