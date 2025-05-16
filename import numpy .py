import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Example dummy input data (replace with real dataset)
X_train = np.random.rand(100, 10)  # 100 samples, 10 timesteps
y_train = np.random.rand(100)  # 100 target values

# Reshape X_train for LSTM (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Ensure y_train is properly shaped
y_train = y_train.reshape((-1, 1))

# Define and compile the model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)
