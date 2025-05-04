from mlxtend.evaluate import bias_variance_decomp 
from sklearn.model_selection import validation_curve
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.evaluate import bias_variance_decomp

from mlxtend.evaluate import bias_variance_decomp
import numpy as np

def get_bias_variance(estimator, X_train, y_train, X_test, y_test):
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    mse, bias, var = bias_variance_decomp(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        loss='mse',
        random_seed=42
    )
    return mse, bias, var

def plot_hyper_param_train_validation_curve(estimator, param_grid, X, y, cv=10, scoring='recall_macro', modelname='model'):
    # iterate over the parameters and get the key and value pairs
    for param, value in param_grid.items():
        # Calculate training and validation scores for different values of max_depth
        train_scores, valid_scores = validation_curve(estimator, X, y,param_name=param, param_range=value,cv=cv, scoring=scoring)
        # Calculate the mean and standard deviation of the training and validation scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        valid_mean = np.mean(valid_scores, axis=1)
        valid_std = np.std(valid_scores, axis=1)

        # Plot the bias-variance tradeoff

        plt.plot(value, train_mean, label='Training score', color='blue')
        plt.fill_between(value, train_mean - train_std,train_mean + train_std, alpha=0.2, color='blue')
        plt.plot(value, valid_mean, label='Cross-validation score', color='red')
        plt.fill_between(value, valid_mean - valid_std,valid_mean + valid_std, alpha=0.2, color='red')
        plt.legend()
        plt.xlabel(param)
        plt.ylabel(scoring)
        plt.title(f'Bias-Variance Tradeoff for {param} using {modelname}')
        plt.show()
