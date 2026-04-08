import pandas as pd
import numpy as np
import random

from Tree_classification import *

class MyForestClf():
    def __init__(self, 
                 n_estimators=10,
                 max_features=0.5,
                 max_samples=0.5,
                 max_depth=5,
                 min_samples_split=2,
                 max_leafs=20,
                 bins=16,
                 criterion='entropy',
                 random_state=42):
        # Параметры леса
        self.n_estimators = n_estimators        # количество деревьев в лесу
        self.max_features = max_features        # доля фичей для каждого дерева. [0.0; 1.0]
        self.max_samples = max_samples          # доля сэмплов для каждого дерева. [0.0; 1.0]
        self.leafs_cnt = 0                      # число листьев в лесу

        # Параметры деревьев решений
        self.max_depth = max_depth              # максимальная глубина деревьев
        self.min_samples_split = min_samples_split  # минимальное количество объектов в узле
        self.max_leafs = max_leafs              # максимальное количество листьев
        self.bins = bins                        # количество бинов гистограммы
        self.criterion = criterion              # эвристика для поиска наилучшего разделения. 'entropy' или 'gini'

        # Прочие параметры
        self.random_state = random_state

    def __str__(self):
        discription = []

        for parameter, value in self.__dict__.items():
            discription.append(f"{parameter}={value}")

        return f"MyForestClf class: {', '.join(discription)}"
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        features = X.columns.to_list()

        # Количество фич и сэмплов для построения дерева
        cols_cnt = round(self.max_features * X.shape[1])
        rows_cnt = round(self.max_samples * X.shape[0])

        self.trees = [None] * self.n_estimators             # Список деревьев
        
        random.seed(self.random_state)
        for tree_number in range(self.n_estimators):
            # Отбор фичей и сэмплов для дерева
            cols_names = random.sample(features, cols_cnt)
            rows_idx = random.sample(range(X.shape[0]), rows_cnt)

            tree = MyTreeClf(max_depth=self.max_depth, 
                             min_samples_split=self.min_samples_split, 
                             max_leafs=self.max_leafs, 
                             bins=self.bins, 
                             criterion=self.criterion)
            tree.fit(X.iloc[rows_idx][cols_names], y.iloc[rows_idx])

            self.trees[tree_number] = tree
            self.leafs_cnt += tree.leafs_cnt

    def predict(self, X: pd.DataFrame, type: str='mean') -> pd.Series:
        # type = 'mean' - усредение показаний деревьев
        # type = 'vote' - наиболее вероятное предсказание деревьев
        if type == 'mean':
            prediction = self.predict_proba(X)
        elif type == 'vote':
            prediction = self.predict_label(X)
        return np.where(prediction > 0.5, 1, 0)
            
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        X = X.reset_index()

        prediction = np.zeros(X.shape[0])
        for tree in self.trees:
            prediction += tree.predict_proba(X)

        return prediction / self.n_estimators
    
    def predict_label(self, X: pd.DataFrame) -> pd.Series:
        X = X.reset_index()

        prediction = np.zeros(X.shape[0])
        for tree in self.trees:
            prediction += tree.predict(X)
        
        return prediction / self.n_estimators


