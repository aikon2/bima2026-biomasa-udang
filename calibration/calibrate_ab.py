import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("calibration_data.csv")

# Log transform
X = np.log(df["length_cm"].values).reshape(-1,1)
y = np.log(df["weight_g"].values)

# Linear regression
model = LinearRegression()
model.fit(X, y)

b = model.coef_[0]
a = np.exp(model.intercept_)

# Goodness of fit
r2 = model.score(X, y)

print("=== Calibration Result ===")
print(f"a = {a:.5f}")
print(f"b = {b:.3f}")
print(f"R^2 = {r2:.3f}")
with open("/../config/biomass.yaml", "w") as f:
    f.write(f"a: {a:.5f}\n")
    f.write(f"b: {b:.3f}\n")