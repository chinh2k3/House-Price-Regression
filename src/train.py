import pandas as pd
from sklearn.model_selection import train_test_split
from processing import Preprocessing
from model import model_train
from scaler import SelectiveScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import joblib
import os

df = pd.read_csv('Data/raw/train.csv')

# Separate independent and dependent variables
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

# Initialize and apply preprocessing
preprocessor = Preprocessing(drop_cols=['Id'])

# Split the data into training and testing sets
X_raw_train, X_raw_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing to the raw data
X_train_pre = preprocessor.fit_transform(X_raw_train)
X_test_pre = preprocessor.transform(X_raw_test)

# StandardScaler for numerical columns
scaler = SelectiveScaler()

X_train = scaler.fit_transform(X_train_pre)
X_test = scaler.transform(X_test_pre)

# Train the model
best_model, best_params, best_score = model_train(X_train, y_train)

print("Best Parameters:", best_params)
print("Best Cross-Validation Score:", best_score)

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)
r2_test = r2_score(y_test, y_pred)
print("Test R2 Score:", r2_test)

train_r2 = best_model.score(X_train, y_train)
test_r2 = best_model.score(X_test, y_test)
print("Score train:", train_r2)
print("Score test:", test_r2)

# Display min and max values for actual and predicted values
print(f'Thực tế min:{y_test.min()}, Thực tế max:{y_test.max()}')
print(f'Dự đoán min:{y_pred.min()}, Dự đoán max:{y_pred.max()}')

# Metrics: MAE and RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Test MAE:", mae)
print("Test RMSE:", rmse)


os.makedirs("artifacts", exist_ok=True)

joblib.dump(scaler, "artifacts/scaler.pkl")
joblib.dump(preprocessor, "artifacts/preprocessor.pkl")
joblib.dump(best_model, "artifacts/model.pkl")

print("Artifacts saved successfully!")
