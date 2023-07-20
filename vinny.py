import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load the data into a DataFrame
data = {
    'Performance': ['Performance Analysis'] * 9,
    'Server': ['Prjct-ANALYTICS-01'] * 9,
    'Date': [
        '2023-01-02T22:30:00', '2023-01-02T16:30:00', '2023-01-03T03:30:00',
        '2023-01-03T00:30:00', '2023-01-01T12:30:00', '2023-01-04T02:30:00',
        '2023-01-03T14:30:00', '2023-01-04T12:30:00', '2023-01-04T10:30:00'
    ],
    'Utilization': [
        0.073611667, 75.38638878, 0.255000836,
        0.228334169, 0.237169167, 0.091419166,
        0.073881668, 0.078333334, 0.082666
    ],
    'Parameter': ['CPU'] * 9
}

df = pd.DataFrame(data)

# Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' as the DataFrame index for time-series analysis
df.set_index('Date', inplace=True)

# Sort the DataFrame by date
df.sort_index(inplace=True)

# Normalize the 'Utilization' values to a range between 0 and 1
scaler = MinMaxScaler()
df['Utilization'] = scaler.fit_transform(df[['Utilization']])

# Prepare the data for LSTM training
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Define the sequence length (number of time steps to look back)
sequence_length = 3

# Create sequences and targets for training
X, y = create_sequences(df['Utilization'].values, sequence_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the LSTM model with dropout regularization
model = Sequential()
model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(sequence_length, 1)))
model.add(Dropout(0.2))
model.add(LSTM(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Add early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.1, callbacks=[early_stopping])

# Evaluate the model on the test set
mse = model.evaluate(X_test, y_test)
print(f"Mean Squared Error on Test Set: {mse}")

# Make predictions on the test set
y_pred = model.predict(X_test)

# Inverse transform the scaled values to get the actual 'Utilization' values
y_pred = scaler.inverse_transform(y_pred)
y_test = y_test.reshape(-1, 1)  # Reshape y_test to a 2D array
y_test = scaler.inverse_transform(y_test)

# Plot the original data and the predicted values
plt.figure(figsize=(10, 6))
plt.plot(df.index[-len(y_test):], y_test, label='True Values', color='blue')
plt.plot(df.index[-len(y_test):], y_pred, label='Predictions', color='red')
plt.xlabel('Date')
plt.ylabel('Utilization')
plt.title('Server Utilization Forecast (LSTM)')
plt.legend()
plt.show()
