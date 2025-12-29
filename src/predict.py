import pandas as pd
import numpy as np
import joblib
import os

DATA_PATH = "Data/raw/test.csv"
ARTIFACT_DIR = "artifacts"
OUTPUT_PATH = "Data/prediction/submission.csv"

preprocessor = joblib.load(os.path.join(ARTIFACT_DIR, "preprocessor.pkl"))
scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
model = joblib.load(os.path.join(ARTIFACT_DIR, "model.pkl"))

df_test = pd.read_csv(DATA_PATH)

if "SalePrice" in df_test.columns:
    df_test = df_test.drop(columns=["SalePrice"])

ids = df_test["Id"] if "Id" in df_test.columns else None

X_test = preprocessor.transform(df_test)
X_test = scaler.transform(X_test)

y_pred = model.predict(X_test)

output = pd.DataFrame({
    "Id": ids,
    "SalePrice": y_pred   
})

output.to_csv(OUTPUT_PATH, index=False)

print(" Prediction completed!")
print(y_pred.min(), y_pred.max())
print(f"Saved to: {OUTPUT_PATH}")
