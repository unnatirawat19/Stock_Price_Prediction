# Stock Price Prediction

## Overview
This project predicts stock prices using machine learning techniques. It utilizes historical stock data to train models that forecast future stock trends.

## Features
- Data collection and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering for better predictions
- Machine learning models for stock price forecasting
- Visualization of stock trends

## Technologies Used
- Python
- NumPy & Pandas
- Matplotlib & Seaborn (for data visualization)
- Scikit-learn (Machine Learning models)
- TensorFlow/Keras (for deep learning models, if applicable)

## Dataset
The dataset includes historical stock market data:
- `Date`: The trading date
- `Open`: Opening price
- `High`: Highest price
- `Low`: Lowest price
- `Close`: Closing price
- `Volume`: Number of shares traded

## Installation
### Prerequisites
Ensure you have Python installed along with the required libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

## Usage
1. Clone the repository:
```bash
git clone https://github.com/your-username/stock-price-prediction.git
cd stock-price-prediction
```
2. Place your dataset (`stock_data.csv`) in the project directory.
3. Run the script:
```bash
python stock_price_prediction.py
```
4. The model will train on historical data and provide predictions.

## Model Training & Evaluation
- The dataset is split into training (80%) and testing (20%).
- Machine learning algorithms such as Linear Regression, Random Forest, and LSTM (for deep learning) are applied.
- Evaluation metrics like RMSE, MAE, and R² score are used.

## Example Prediction
The model takes past stock prices and predicts:
```
Predicted closing price: $120.45
```

## Future Enhancements
- Implement advanced deep learning models (e.g., LSTMs, Transformers).
- Incorporate real-time stock market data.
- Deploy as a web application for live predictions.

## Contributors
- [Unnati Rawat](https://github.com/unnatirawat19)



