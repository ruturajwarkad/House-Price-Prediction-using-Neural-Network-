# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load the dataset
file_path = '/Users/ruturajwarkad/Desktop/NNFL case study/puner.csv'
df = pd.read_csv(file_path)

# Select the relevant features and target
features = ['area', 'bathroom', 'bhk', 'carpetarea']
target = 'price'

# Convert 'area' and 'carpetarea' columns to numeric, removing commas if present
df['area'] = pd.to_numeric(df['area'].str.replace(',', ''), errors='coerce')
df['carpetarea'] = pd.to_numeric(df['carpetarea'], errors='coerce')

# Drop rows with missing values in the selected features
df_clean = df.dropna(subset=features + [target])

# Split data into features (X) and target (y)
X = df_clean[features].values
y = df_clean[target].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a simple neural network model with increased neurons and batch size
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),  # Increased neurons
    Dense(64, activation='relu'),  # Increased neurons
    Dense(32, activation='relu'),  # Increased neurons
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Create a checkpoint callback to save the best model
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')

# Train the model with checkpointing for 500 epochs
history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=500, batch_size=64, verbose=1, callbacks=[checkpoint])

# Function to format price in Indian currency format (xx,xx,xxx)
def format_price_inr(price):
    price_str = str(int(price))  # Convert to int to remove decimals, then to string
    if len(price_str) <= 3:
        return price_str
    elif len(price_str) <= 5:
        return price_str[:-3] + ',' + price_str[-3:]
    else:
        return price_str[:-5] + ',' + price_str[-5:-3] + ',' + price_str[-3:]

# Function to take user input for house preferences
def get_user_input():
    area = float(input("Enter the area (in sq.ft): "))
    bathroom = int(input("Enter the number of bathrooms: "))
    bhk = int(input("Enter the number of BHK: "))
    carpetarea = float(input("Enter the carpet area (in sq.ft): "))
    return [area, bathroom, bhk, carpetarea]

# Function to predict the price based on user input
def predict_price(model, scaler):
    user_input = get_user_input()

    # Scale the user input
    user_input_scaled = scaler.transform([user_input])

    # Predict the price using the trained model
    predicted_price = model.predict(user_input_scaled)

    # Format the predicted price with commas for Indian currency
    formatted_price = format_price_inr(predicted_price[0][0])  # Format as Indian currency
    print(f"Predicted House Price: ₹ {formatted_price}")

# Evaluate the model on test data
loss = model.evaluate(X_test_scaled, y_test)
print(f'Test Loss (MSE): {loss}')

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Calculate additional metrics: RMSE, MAE, R² Score, and MAPE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Root Mean Squared Error (RMSE)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)  # Calculate R² Score
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error (MAPE)

# Print the metrics
print(f'Test RMSE: {rmse}')
print(f'Test MAE: {mae}')
print(f'Test R² Score: {r2}')  # Print R² Score
print(f'Test MAPE: {mape}%')  # Print MAPE percentage

# Predict house prices for a new user input
predict_price(model, scaler)

# Save the trained model to disk
model.save('house_price_model.h5')
