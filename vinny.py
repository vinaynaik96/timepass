import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the data into a DataFrame
data = {
    'Performance': ['Performance Analysis'] * 36,
    'Server': ['Prjct-ANALYTICS-01'] * 36,
    'Date': [
        '2023-01-02T22:30:00', '2023-01-02T16:30:00', '2023-01-03T03:30:00',
        '2023-01-03T00:30:00', '2023-01-01T12:30:00', '2023-01-04T02:30:00',
        '2023-01-03T14:30:00', '2023-01-04T12:30:00', '2023-01-04T10:30:00',
        # Additional 30 data points
        '2023-01-05T03:30:00', '2023-01-06T01:30:00', '2023-01-07T00:30:00',
        '2023-01-08T02:30:00', '2023-01-09T04:30:00', '2023-01-10T06:30:00',
        '2023-01-11T08:30:00', '2023-01-12T10:30:00', '2023-01-13T12:30:00',
        '2023-01-14T14:30:00', '2023-01-15T16:30:00', '2023-01-16T18:30:00',
        '2023-01-17T20:30:00', '2023-01-18T22:30:00', '2023-01-19T00:30:00',
        '2023-01-20T02:30:00', '2023-01-21T04:30:00', '2023-01-22T06:30:00',
        '2023-01-23T08:30:00', '2023-01-24T10:30:00', '2023-01-25T12:30:00',
        '2023-01-26T14:30:00', '2023-01-27T16:30:00', '2023-01-28T18:30:00',
        '2023-01-29T20:30:00', '2023-01-30T22:30:00', '2023-01-31T00:30:00'
    ],
    'Utilization': [
        0.073611667, 75.38638878, 0.255000836,
        0.228334169, 0.237169167, 0.091419166,
        0.073881668, 0.078333334, 0.082666,
        # Additional 30 data points (random values for demonstration purposes)
        0.256419166, 0.09738878, 0.123000836,
        0.548334169, 0.872169167, 0.131419166,
        0.648881668, 0.544333334, 0.673666,
        0.475111667, 0.64899878, 0.891230836,
        0.365334169, 0.331569167, 0.541419166,
        0.276881668, 0.235433334, 0.843666,
        0.956111667, 0.21238878, 0.218000836,
        0.417334169, 0.223169167, 0.991419166,
        0.073881668, 0.578333334, 0.142666
    ],
    'Parameter': ['CPU'] * 36
}

df = pd.DataFrame(data)

# Normalize the 'Utilization' values
scaler = MinMaxScaler()
df['Utilization'] = scaler.fit_transform(df[['Utilization']])

# Convert time series data into supervised learning data
def create_supervised_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

n_steps = 7  # Number of past days to use for prediction

# Create supervised data
X, y = create_supervised_data(df['Utilization'].values, n_steps)

# Split data into training and validation sets
split_ratio = 0.8
split_index = int(split_ratio * len(X))
X_train, X_val = X[:split_index], X[split_index:]
y_train, y_val = y[:split_index], y[split_index:]

# Reshape the input data to fit LSTM format (samples, time steps, features)
X_train = X_train.reshape(-1, n_steps, 1)
X_val = X_val.reshape(-1, n_steps, 1)

# Build the LSTM model
model = keras.Sequential([
    keras.layers.LSTM(64, activation='relu', input_shape=(n_steps, 1)),
    keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
epochs = 100
batch_size = 16
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

# Predict the utilization values on the training and validation data
y_train_pred = model.predict(X_train)
y_train_pred = scaler.inverse_transform(y_train_pred).flatten()

y_val_pred = model.predict(X_val)
y_val_pred = scaler.inverse_transform(y_val_pred).flatten()

# Calculate RMSE for training and validation data
y_train = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_val = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
rmse_train = np.sqrt(np.mean((y_train_pred - y_train)**2))
rmse_val = np.sqrt(np.mean((y_val_pred - y_val)**2))
print("RMSE for Training Data:", rmse_train)
print("RMSE for Validation Data:", rmse_val)


# Forecast next 22 days (15 days + 7 days) with 24 hours per day
forecast_input = df['Utilization'].values[-n_steps:]
forecasted_values = []

for _ in range(22):
    forecast_input_reshaped = forecast_input.reshape(1, n_steps, 1)
    forecast = model.predict(forecast_input_reshaped)[0, 0]
    forecasted_values.append(forecast)
    forecast_input = np.append(forecast_input, forecast)
    forecast_input = forecast_input[-n_steps:]

# Inverse transform the forecasted values to their original scale
forecasted_values = scaler.inverse_transform(np.array(forecasted_values).reshape(-1, 1))

# Create a DataFrame for the forecasted results
forecast_dates = pd.date_range(start=df['Date'].iloc[-1], periods=22, freq='D')
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Utilization': forecasted_values.flatten()})

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Plot the original, predicted, and forecasted utilization data in one graph
plt.figure(figsize=(12, 8))

# Plot the original utilization data
plt.plot(df['Date'], df['Utilization'], label='Original Utilization', color='blue')

# Plot the predicted utilization values on the training data
train_pred_dates = df['Date'][n_steps : n_steps + len(y_train_pred)]
plt.plot(train_pred_dates, y_train_pred, label='Predicted Utilization (Train)', color='red', linestyle='dashed')

# Plot the predicted utilization values on the validation data
val_pred_dates = df['Date'][split_index + n_steps : split_index + n_steps + len(y_val_pred)]
plt.plot(val_pred_dates, y_val_pred, label='Predicted Utilization (Validation)', color='green', linestyle='dashed')

# Plot the forecasted utilization values
plt.plot(forecast_df['Date'], forecast_df['Utilization'], label='Forecasted Utilization', color='blue', marker='o')

plt.xlabel('Date')
plt.ylabel('Utilization')
plt.title('Original Utilization, Predicted Utilization, and Forecasted Utilization')
plt.legend()
plt.grid(True)

# Show the combined graph
plt.show()

# Print the report for the next 7 days' utilization predictions
forecast_report = forecast_df.iloc[15:]
print("Forecasted Utilization for the Next 7 Days:")
print(forecast_report)

# Save the forecast report to an Excel file
forecast_report.to_excel('forecast_report.xlsx', index=False)
