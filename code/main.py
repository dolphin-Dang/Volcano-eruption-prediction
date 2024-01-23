import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from network import get_model
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def calculate_accuracy(true_values, predicted_values, threshold):
    errors = np.abs(true_values - predicted_values) / true_values
    correct_predictions = np.sum(errors <= threshold)
    accuracy = correct_predictions / len(true_values) * 100
    return accuracy

if __name__ == "__main__":
    data = pd.read_csv('data/standard_merged_data.csv')

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # train:test = 8:2
    X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = ['linear', 'SVM', 'DecisionTree', 'RandomForest', 'MLP']
    model_choice = 4
    threshold = 0.3
    print(f"Model: {models[model_choice]}; Threshold: {threshold}")

    model = get_model(models[model_choice])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    print("MSE:", mse)

    acc = calculate_accuracy(y_test, pred, threshold)
    print(f"Prediction accuracy: {acc}")

    pred_train = model.predict(X_train)
    acc_train = calculate_accuracy(y_train, pred_train, threshold)
    print(f"Train accuracy: {acc_train}")
    
    # remove outlier
    outlier_index = np.argmax(np.abs(pred - y_test))
    X_test_filtered = X_test.drop(X_test.index[outlier_index])
    y_test_filtered = y_test.drop(y_test.index[outlier_index])
    pred_filtered = np.delete(pred, outlier_index)

    # sort by pred
    sorted_indices = pred_filtered.argsort()
    X_test_sorted = X_test_filtered.iloc[sorted_indices]
    y_test_sorted = y_test_filtered.iloc[sorted_indices]
    pred_sorted = pred_filtered[sorted_indices]

    # visualization
    plt.scatter(range(len(y_test_sorted)), y_test_sorted, color='b', label='True')
    plt.plot(range(len(pred_sorted)), pred_sorted, color='r', label='Regression')
    plt.xlabel('Data Points')
    plt.ylabel('Target')
    plt.title('Comparison of Predictions and True Values')
    plt.legend()
    plt.show()
