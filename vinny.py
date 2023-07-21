# Create a DataFrame for the forecasted results
forecast_dates = pd.date_range(start=df['Date'].iloc[-1], periods=15, freq='D')
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Utilization': forecasted_values.flatten()})

# Plot the original, predicted, and forecasted utilization data in one graph
plt.figure(figsize=(12, 8))

# Plot the original utilization data
plt.plot(df.index, df['Utilization'], label='Original Utilization', color='blue')

# Plot the predicted utilization values on the training data
train_pred_dates = df.index[n_steps : n_steps + len(y_train_pred)]
plt.plot(train_pred_dates, y_train_pred, label='Predicted Utilization (Train)', color='red', linestyle='dashed')

# Plot the predicted utilization values on the validation data
val_pred_dates = df.index[split_index + n_steps : split_index + n_steps + len(y_val_pred)]
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
