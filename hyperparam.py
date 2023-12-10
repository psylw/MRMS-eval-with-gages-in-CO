import scipy.stats as stats
import numpy as np
from scipy.stats import uniform, randint


param = [
    # knn
    {'n_neighbors': randint(1, 50), 
     'weights': ['uniform', 'distance'], 'leaf_size':np.arange(0,60,5) ,'p': [1, 2]},
    # dt
    {'max_depth': randint(1, 20),'min_samples_split':randint(2, 20),'min_samples_leaf':randint(1, 20),
    
     },


    # RF
    {"n_estimators": randint(50, 250), 
     
                                 "max_depth":randint(5, 50),
                                          "min_samples_split":randint(2, 20),
                                         "min_samples_leaf":randint(1, 20),
                                         'max_features': ['auto', 'sqrt'],
                                         'bootstrap': [True, False]},
    # bagging
    {"n_estimators": randint(50, 200),
     'max_samples':uniform(0.5, 0.5),
     'max_features':uniform(0.5, 0.5)},
    # MLP
    {
    'hidden_layer_sizes': randint(50, 500),  # Number of neurons in each hidden layer
    'activation': ['relu', 'tanh', 'logistic'],  # Activation function
    'solver': ['adam', 'sgd'],  # Solver for weight optimization
    'alpha': uniform(0.0001, 0.01),  # L2 penalty (regularization term)
    'learning_rate': ['constant', 'invscaling', 'adaptive'],  # Learning rate schedule
    'learning_rate_init': uniform(0.001, 0.1),  # Initial learning rate
    'max_iter': randint(100, 1000),  # Maximum number of iterations
    },
    #Ada
    {'n_estimators': randint(50, 500), 
     'learning_rate':uniform(0.01, 1.0)},
    

    #xgb
    {
    'n_estimators': randint(100, 1000),  # Number of trees
    'learning_rate': uniform(0.01, 0.3),  # Learning rate
    'max_depth': randint(3, 10),  # Maximum depth of a tree
    'min_child_weight': randint(1, 10),  # Minimum sum of instance weight needed in a child
    'subsample': uniform(0.6, 0.4),  # Subsample ratio of the training instance
    'colsample_bytree': uniform(0.6, 0.4),  # Subsample ratio of columns when constructing each tree
    'gamma': uniform(0, 0.5),  # Minimum loss reduction required to make a further partition on a leaf node of the tree
    'reg_alpha': uniform(0, 1),  # L1 regularization term on weights
    'reg_lambda': uniform(0, 1)  # L2 regularization term on weights
    },
    #gbc
    {
    'n_estimators': randint(50, 200),  # Number of boosting stages to be run
    'learning_rate': uniform(0.01, 0.5),  # Learning rate
    'max_depth': randint(3, 10),  # Maximum depth of the individual trees
    'min_samples_split': randint(2, 20),  # Minimum samples required to split a node
    'min_samples_leaf': randint(1, 20),  # Minimum samples required at each leaf node
    'max_features': ['auto', 'sqrt', 'log2', None]  # Number of features to consider at each split
    },
    # SVC
    {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel types to choose from
    'C': uniform(loc=0, scale=10),  # Regularization parameter C (uniform distribution between 0 and 10)
    'gamma': ['scale', 'auto'],  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
    'degree': randint(2, 6),  # Degree of the polynomial kernel (random integer between 2 and 5)
    'epsilon': uniform(loc=0, scale=1),  # Epsilon in the SVR model (uniform distribution between 0 and 1)
    }
]
