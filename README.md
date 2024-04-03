# stock-price-prediction-by-ML

This is FIN3210 Final Project in CUHK(SZ) on Fall 2022.
Our objective is to apply machine learning and deep learning models to predicting stocks' return.
We start with downloading Alpha158 dataset.
Then, we try different models and build strategies to see their backtest performance.
Finally, we select a bunch of models to conduct a strategy-based portfolio.

Given the complex nature and the breadth of the code across multiple files, I'll provide a high-level overview and some key points about each script to help explain their functionality and purpose. 

### Overview

The codebase appears to be part of a financial data analysis and machine learning project aimed at stock market predictions. It incorporates various machine learning models for time series forecasting and portfolio optimization based on stock price data. Key components include data preprocessing, model training and evaluation, portfolio construction, and performance backtesting.

### ALSTM.py

This file defines an Attention-based Long Short-Term Memory (ALSTM) model class. The `ALSTMModel` class extends PyTorch's `nn.Module` and is designed for time series forecasting. It incorporates LSTM layers for sequential data processing, supplemented with an attention mechanism to focus on important time steps. The model takes features of dimensions `d_feat` and predicts outcomes based on the learned patterns.

### main.py

The `main.py` file orchestrates the overall workflow, including data preparation, model training, evaluation, and generating trading signals. It utilizes models defined in other files (like ALSTM, Transformer, and Temporal Convolutional Network) and applies them to financial time series data. This script handles the training of both machine learning and deep learning models, backtesting of strategies, and analysis of results through metrics like Mean Squared Error (MSE) and Information Coefficient (IC).

### ML_Model.py

This script defines several machine learning models for regression tasks, including Ridge Regression, Lasso Regression, Random Forest, Adaboost, XGBoost, LightGBM, and CatBoost. Each model class provides methods for setting parameters, fitting the model to data, performing cross-validation, and hyperparameter optimization using `hyperopt`. It serves as a utility for experimenting with different regression techniques on financial data.

### Portfolio.py

This file focuses on portfolio construction and optimization based on the predictions from various models. It defines functions for calculating optimal asset allocations using the mean-variance optimization approach, adjusting for risk tolerance through the parameter gamma. The script utilizes `cvxopt` for solving quadratic programming problems inherent in portfolio optimization.

### TCN.py

The `TCN.py` file defines a Temporal Convolutional Network (TCN) model for time series forecasting. TCNs are effective for capturing long-term dependencies in data thanks to dilated convolutions. The script sets up a TCN architecture comprising several layers of dilated causal convolutions, designed to predict future stock prices or returns based on past data.

### Transformer.py

This script implements a Transformer model tailored for time series forecasting in the stock market. Transformers use self-attention mechanisms to weigh the importance of different parts of the input data. The model defined in this file is adapted for sequential data processing, aiming to capture complex dependencies in financial time series.

### Explanation of Code Functionality

- **Data Preprocessing**: The codebase includes functionalities for handling financial time series data, filling missing values, and preparing datasets for model training.
- **Model Training and Evaluation**: Various regression and deep learning models are trained on historical stock data. The models are evaluated based on their predictive accuracy using metrics like MSE and IC.
- **Portfolio Optimization**: Based on the predictions from trained models, the portfolio optimization component calculates the optimal asset weights to maximize returns for a given level of risk.
- **Backtesting**: The strategies derived from model predictions are backtested to assess their performance in historical scenarios, helping validate their effectiveness in real-world trading conditions.

Overall, the code represents a comprehensive approach to applying advanced machine learning techniques for financial forecasting and investment strategy development.
