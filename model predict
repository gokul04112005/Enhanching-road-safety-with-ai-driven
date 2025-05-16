from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# X: features, y: target
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Step 1: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: (Optional) Scale target if you're planning to inverse_transform later
scaler = MinMaxScaler()
y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))

# Step 3: Train the model
model = LinearRegression()
model.fit(X_train, y_train_scaled)

# Step 4: Predict
y_pred = model.predict(X_test)

# Step 5: Inverse transform predictions and y_test
if y_pred.ndim == 1:
    y_pred = y_pred.reshape(-1, 1)

y_pred = scaler.inverse_transform(y_pred)

if y_test.ndim == 1:
    y_test_actual = y_test.reshape(-1, 1)
else:
    y_test_actual = y_test

y_test_actual = scaler.inverse_transform(y_test_actual)

# Step 6: Evaluation
print("RMSE:", np.sqrt(mean_squared_error(y_test_actual, y_pred)))
print("R² Score:", r2_score(y_test_actual, y_pred))
