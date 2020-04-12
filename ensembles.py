import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize_scalar
import time


class RandomForestMSE:
    def __init__(self, n_estimators, max_depth=None, feature_subsample_size=None,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        
        feature_subsample_size : int
            The size of feature set for each tree.
            If None feature_subsample_size = n_features
        """
        self.feature_subsample_size = feature_subsample_size

        self.trees = [DecisionTreeRegressor(max_depth=max_depth, **trees_parameters) for i in range(n_estimators)]

    def fit(self, X, y):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        y : numpy ndarray
            Array of size n_objects
        """
        np.random.seed(42)
        self.feature_list = []
        
        if self.feature_subsample_size == None:
            self.feature_subsample_size = X.shape[1]
        
        start_time = time.time()
        
        for tree in self.trees:
            feature_indexes = self.get_feature_indexes(X.shape[1])
            sample_indexes = self.get_sample_indexes(X.shape[0])
            self.feature_list.append(feature_indexes)
            tree.fit(X[sample_indexes, :][:, feature_indexes], y[sample_indexes])
        
        return time.time() - start_time

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        predict_list = [tree.predict(X[:, self.feature_list[i]]) for i, tree in enumerate(self.trees)]
        return np.mean(predict_list, axis=0)
    
    def get_sample_indexes(self, size):
        """
        size: int
            The 0-dimention size of matrix
                            <= X_train.shape[0]
        
        Returns
        -------
        random_indexes : numpy ndarray
            Array of size feature_subsample_size
        """
        return np.random.choice(size, np.random.randint(size // 24, size // 2), replace=False)
    
    def get_feature_indexes(self, size):
        """
        size: int
            The 1-dimention size of matrix
            X_train.shape[1]
        
        Returns
        -------
        random_indexes : numpy ndarray
            Array of size feature_subsample_size
        """
        return np.random.choice(size, self.feature_subsample_size, replace=False)

class GradientBoostingMSE:
    def __init__(self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        
        learning_rate : float
            F_m = F_m-1 + learning_rate * c_m * f_m
        
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        
        feature_subsample_size : int
            The size of feature set for each tree.
            If None feature_subsample_size = n_features
        """
        self.feature_subsample_size = feature_subsample_size
        self.learning_rate = learning_rate
        self.trees = [DecisionTreeRegressor(max_depth=max_depth, **trees_parameters) for i in range(n_estimators)]
        
    def fit(self, X, y):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        y : numpy ndarray
            Array of size n_objects
            
        F_m = F_m_old + learning_rate * c_m * f_m
        """
        if self.feature_subsample_size == None:
            self.feature_subsample_size = X.shape[1]
        
        np.random.seed(42)
        
        F_m = 0
        self.coef = []
        self.feature_list = []
        start_time = time.time()
        
        for tree in self.trees:
            feature_indexes = self.get_feature_indexes(X.shape[1])
            sample_indexes = self.get_sample_indexes(X.shape[0])
            self.feature_list.append(feature_indexes)
            
            tree.fit(X[sample_indexes, :][:, feature_indexes], (y - F_m)[sample_indexes])
            f_m = tree.predict(X[:, feature_indexes])
            best_coef = minimize_scalar(lambda c: 
                                         self.mean_squared_error(y, F_m + c * f_m))
            self.coef.append(best_coef.x)
            F_m += self.learning_rate * best_coef.x * f_m
        
        return time.time() - start_time

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        np.random.seed(42)
        out = 0
        
        for i, tree in enumerate(self.trees):
            out += self.learning_rate * self.coef[i] * tree.predict(X[:, self.feature_list[i]])
        
        return out

    def mean_squared_error(self, y, ens):
        return np.mean((y - ens) ** 2)

    def get_sample_indexes(self, size):
        """
        size: int
            The 0-dimention size of matrix
                            <= X_train.shape[0]
        
        Returns
        -------
        random_indexes : numpy ndarray
            Array of size feature_subsample_size
        """
        return np.random.choice(size, np.random.randint(size // 24, size // 2), replace=False)

    def get_feature_indexes(self, size):
        """
        size: int
            The 1-dimention size of matrix
            X_train.shape[1]
        
        Returns
        -------
        random_indexes : numpy ndarray
            Array of size feature_subsample_size
        """
        return np.random.choice(size, self.feature_subsample_size, replace=False)
