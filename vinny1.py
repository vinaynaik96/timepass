import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Synthetic dataset
np.random.seed(42)
data = np.random.rand(200, 1)

# Normalize the data
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Create sequences and targets
sequence_length = 30  # Number of time steps to look back
X, y = [], []
for i in range(len(data_normalized) - sequence_length):
    X.append(data_normalized[i : i + sequence_length])
    y.append(data_normalized[i + sequence_length])

X = np.array(X)
y = np.array(y)

# Split the dataset into training and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Define the ModelCheckpoint callback to save the best model
checkpoint_callback = ModelCheckpoint(
    filepath='best_model.h5',
    monitor='val_loss',  # Monitoring validation loss for saving the best model
    save_best_only=True,  # Save only the best model
    save_weights_only=False,  # Save entire model
    verbose=1
)

# Define EarlyStopping callback to stop training if there is no improvement in val_loss
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,  # Number of epochs with no improvement to wait before stopping
    restore_best_weights=True
)

# Train the model with the ModelCheckpoint and EarlyStopping callbacks
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint_callback, early_stopping_callback]  # Add the callbacks to the training process
)

# Make predictions for the next 15 days using persistence model
future_days = 15
future_predictions = []
last_observed_value = X_test[-1][-1][0]

for _ in range(future_days):
    future_predictions.append(last_observed_value)

# Inverse transform persistence predictions to the original scale
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Make predictions for the test set
test_predictions = model.predict(X_test)
test_predictions = scaler.inverse_transform(test_predictions)

# Make predictions for the training set (for demonstration purposes)
train_predictions = model.predict(X_train)
train_predictions = scaler.inverse_transform(train_predictions)

# Inverse transform the original data
original_data = scaler.inverse_transform(data_normalized)

# Create a date range for the future predictions (for demonstration purposes)
date_range = pd.date_range(start='2023-07-24', periods=len(original_data))

# Plot the original data, train data predictions, test data predictions, and future predictions
plt.figure(figsize=(12, 6))
plt.plot(date_range, original_data, label='Original Data', color='blue')
plt.plot(date_range[sequence_length:train_size], train_predictions, label='Train Data Predictions', color='green')
plt.plot(date_range[train_size + sequence_length:], test_predictions, label='Test Data Predictions', color='orange')
plt.plot(pd.date_range(start=date_range[-1], periods=future_days + 1)[1:], future_predictions, label='Future Predictions', color='red')

plt.xlabel('Date')
plt.ylabel('Value')
plt.title('LSTM Time Series Forecasting')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# ...

# Inverse transform the original data
original_data = scaler.inverse_transform(data_normalized)

# Create a date range for the future predictions (for demonstration purposes)
date_range = pd.date_range(start='2023-07-24', periods=len(original_data))

# Create date range for train data predictions
train_date_range = pd.date_range(start=date_range[sequence_length], periods=train_size)

# Plot the original data, train data predictions, test data predictions, and future predictions
plt.figure(figsize=(12, 6))
plt.plot(date_range, original_data, label='Original Data', color='blue')
plt.plot(train_date_range, train_predictions, label='Train Data Predictions', color='green')
plt.plot(date_range[train_size + sequence_length:], test_predictions, label='Test Data Predictions', color='orange')
plt.plot(pd.date_range(start=date_range[-1], periods=future_days + 1)[1:], future_predictions, label='Future Predictions', color='red')

plt.xlabel('Date')
plt.ylabel('Value')
plt.title('LSTM Time Series Forecasting')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Calculate evaluation metrics for the test set predictions
y_test_pred = best_model.predict(X_test)
y_test_pred = scaler.inverse_transform(y_test_pred)
y_test_actual = scaler.inverse_transform(y_test)

mse_test = mean_squared_error(y_test_actual, y_test_pred)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test_actual, y_test_pred)
r2_test = r2_score(y_test_actual, y_test_pred)

print("Test Set Evaluation Metrics:")
print(f"MSE: {mse_test:.4f}")
print(f"RMSE: {rmse_test:.4f}")
print(f"MAE: {mae_test:.4f}")
print(f"R2 Score: {r2_test:.4f}")
