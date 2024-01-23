import sklearn
import sklearn.tree
import sklearn.ensemble
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

def get_model(name):
    if name == 'linear':
        model = LinearRegression()
    elif name == 'SVM':
        model = sklearn.svm.SVR()
    elif name == 'DecisionTree':
        model = sklearn.tree.DecisionTreeRegressor()
    elif name == 'RandomForest':
        model = sklearn.ensemble.RandomForestRegressor(n_estimators=30)

    elif name == 'MLP':
        model = MLPRegressor(   
                hidden_layer_sizes=(50, 50), 
                activation='relu', 
                solver='adam',
                alpha=0.01, 
                batch_size='auto', 
                learning_rate='constant', 
                learning_rate_init=0.001, 
                max_iter=200
        )
    return model