
import pandas as pd, numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("evaluation/data.csv")
mae = mean_absolute_error(df.weight_gt, df.weight_pred)
mape = np.mean(abs((df.weight_gt-df.weight_pred)/df.weight_gt))*100
r2 = r2_score(df.weight_gt, df.weight_pred)
print(f"MAE:{mae:.3f}g  MAPE:{mape:.2f}%  R2:{r2:.3f}")
