"""
Assignment 6 Part 1: Student Performance Prediction
Name: Aidan
Date: 11/21/2025

This assignment predicts student test scores based on hours studied.
Complete all the functions below following the in-class ice cream example.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def load_and_explore_data(filename):
    """
    Load the student scores data and explore it
    
    Args:
        filename: name of the CSV file to load
    
    Returns:
        pandas DataFrame containing the data
    """
    # load csv
    data = pd.read_csv(filename)
    
    # basic prints
    print(data.head())
    print(data.shape)
    print(data.describe())
    
    return data


def create_scatter_plot(data):
    """
    Create a scatter plot to visualize the relationship between hours studied and scores
    
    Args:
        data: pandas DataFrame with Hours and Scores columns
    """
    plt.figure(figsize=(10, 6))
    
    # plotting hours vs scores
    plt.scatter(data['Hours'], data['Scores'], color='purple', alpha=0.6)
    
    plt.xlabel('Hours Studied')
    plt.ylabel('Test Score')
    plt.title('Student Test Scores vs Hours Studied')
    
    plt.grid(alpha=0.3)
    plt.savefig('scatter_plot.png', dpi=300)
    plt.show()


def split_data(data):
    """
    Split data into features (X) and target (y), then into training and testing sets
    
    Args:
        data: pandas DataFrame with Hours and Scores columns
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # double brackets for dataframe
    X = data[['Hours']]
    y = data['Scores']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"training set size: {len(X_train)}")
    print(f"testing set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Create and train a linear regression model
    
    Args:
        X_train: training features
        y_train: training target values
    
    Returns:
        trained LinearRegression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print(f"slope: {model.coef_[0]}")
    print(f"intercept: {model.intercept_}")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance on test data
    
    Args:
        model: trained LinearRegression model
        X_test: testing features
        y_test: testing target values
    
    Returns:
        predictions array
    """
    predictions = model.predict(X_test)
    
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    print(f"r2 score: {r2}")
    print(f"mean squared error: {mse}")
    print(f"root mean squared error: {rmse}")
    
    return predictions


def visualize_results(X_train, y_train, X_test, y_test, predictions, model):
    """
    Visualize the model's predictions against actual values
    
    Args:
        X_train: training features
        y_train: training target values
        X_test: testing features
        y_test: testing target values
        predictions: model predictions on test set
        model: trained model (to plot line of best fit)
    """
    plt.figure(figsize=(12, 6))
    
    plt.scatter(X_train, y_train, color='blue', label='Training Data')
    plt.scatter(X_test, y_test, color='green', label='Test Data (Actual)')
    plt.scatter(X_test, predictions, color='red', marker='x', label='Predictions')
    
    # line of best fit
    x_range = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    y_range = model.predict(x_range)
    plt.plot(x_range, y_range, color='black')
    
    plt.xlabel('Hours Studied')
    plt.ylabel('Test Score')
    plt.title('Student Test Scores vs Hours Studied (Predictions)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.savefig('predictions_plot.png', dpi=300)
    plt.show()


def make_prediction(model, hours):
    """
    Make a prediction for a specific number of hours studied
    
    Args:
        model: trained LinearRegression model
        hours: number of hours to predict score for
    
    Returns:
        predicted test score
    """
    hours_arr = np.array([[hours]])
    prediction = model.predict(hours_arr)[0]
    
    print(f"prediction for {hours} hours: {prediction}")
    
    return prediction


if __name__ == "__main__":
    print("=" * 70)
    print("STUDENT PERFORMANCE PREDICTION")
    print("=" * 70)
    
    # 1. load
    data = load_and_explore_data('student_scores.csv')
    
    # 2. visualize
    create_scatter_plot(data)
    
    # 3. split
    X_train, X_test, y_train, y_test = split_data(data)
    
    # 4. train
    model = train_model(X_train, y_train)
    
    # 5. evaluate
    predictions = evaluate_model(model, X_test, y_test)
    
    # 6. visualize results
    visualize_results(X_train, y_train, X_test, y_test, predictions, model)
    
    # 7. predict
    make_prediction(model, 7)
    
    print("\n" + "=" * 70)
    print("done")
    print("=" * 70)
