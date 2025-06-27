🏠 **House Price Prediction using Neural Network**

This project predicts house prices in Pune using a neural network model built with TensorFlow and Keras. The model is trained on features like area, number of bathrooms, BHK, and carpet area.

**📌 Features**

Predicts house prices using a feedforward neural network.
User-friendly command-line interface for custom input.
Model trained with:
Scaled input features
Model checkpointing
Evaluation metrics: RMSE, MAE, R², MAPE
Trained model saved as .h5 file for future use.


**🧠 Technologies Used**

Python
Pandas
NumPy
scikit-learn
TensorFlow / Keras


**📊 Dataset**

Custom CSV dataset of Pune houses (puner.csv)
Required columns: area, bathroom, bhk, carpetarea, price


**📁 Project Structure**

house_price_prediction/
├── puner.csv
├── house_price_model.h5
├── best_model.keras
└── main.py  # This script contains the full code


**🛠️ How it Works**

Data Preprocessing
Cleans and standardizes data.
Splits into train and test sets.
Applies feature scaling.
Model Architecture
3 hidden layers (128, 64, 32 neurons).
ReLU activation functions.
Trained using adam optimizer and MSE loss.
Training
500 epochs with batch size 64.
ModelCheckpoint saves the best model.
Evaluation Metrics
RMSE (Root Mean Square Error)
MAE (Mean Absolute Error)
R² Score
MAPE (Mean Absolute Percentage Error)
Prediction Interface
User can enter house features via input.
Model predicts price and formats it in INR style.


**🚀 How to Run**

Clone the repository.
Make sure you have the dataset: puner.csv.
Run the script:
python main.py
Follow the prompts to predict house prices.


**📈 Sample Output**

Test RMSE: 1254321.45
Test MAE: 854321.67
Test R² Score: 0.87
Test MAPE: 12.34%
Predicted House Price: ₹ 64,25,000


**📦 Future Improvements**

Add GUI for better user interaction
Use more features like location, amenities, etc.
Deploy using Flask or Streamlit


**👨‍💻 Author**

Ruturaj Warkad
B.Tech in Computer Engineering
Project for Neural Network Case Study

