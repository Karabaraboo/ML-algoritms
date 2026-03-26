import numpy as np
import pandas as pd
import random
from Tree_regression import *

class MyForestReg():
    def __init__(self,
                 n_estimators: int=10,
                 max_features: float=0.5,
                 max_samples: float=0.5,
                 random_state: int=42,
                 max_depth: int=5,
                 min_samples_split: int=2,
                 max_leafs: int=20,
                 bins: int=16,
                 verbose=False):
        # Параметры леса
        self.n_estimators = n_estimators            # Количество деревьев в лесу
        self.max_features = max_features            # Доля фичей для каждого дерева [0, 1]
        self.max_samples = max_samples              # Доля сэмплов для кажого дерева [0, 1]
        # Параметры дерева
        self.max_depth = max_depth                  # Максимальная глубина
        self.min_samples_split = min_samples_split  # Минимальное число сэмплов в узле
        self.max_leafs = max_leafs                  # Максимальное число листов
        self.bins = bins                            # Число бинов при построении гистограммы для определения разделителей
        
        self.random_state = random_state            # фиксация сида

        # Прочие атрибуты класса
        self.leafs_cnt = 0                          # Количество листьев во всём лесу
        self.trees = None                           # Сохранение деревьев решений

        self.verbose = verbose

    def __str__(self):
        parameters = []

        for name, value in self.__dict__.items():
            parameters.append(f"{name}={value}")

        return f"MyForestReg class: {', '.join(parameters)}"
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Количество фичей и элементов для отбора
        cols_cnt = round(self.max_features * X.shape[1])
        rows_cnt = round(self.max_samples * X.shape[0])

        self.trees = [None] * self.n_estimators      # Список для деревьев
        random.seed(self.random_state)
        
        for tree_number in range(self.n_estimators):
            cols_idx = random.sample(X.columns.to_list(), cols_cnt)
            rows_idx = random.sample(range(rows_cnt), rows_cnt)

            # Обучение дерева
            tree = MyTreeReg(max_depth=self.max_depth,
                                     min_samples_split=self.min_samples_split,
                                     max_leafs=self.max_leafs,
                                     bins=self.bins)
            tree.fit(X.loc[X.index[rows_idx], cols_idx], y.loc[y.index[rows_idx]])

            self.leafs_cnt += tree.leafs_cnt            
            self.trees[tree_number] = tree

            if self.verbose:
                print(f"cols_cnt={cols_cnt}, rows_cnt={rows_cnt},\ncols_idx={cols_idx},\nrows_idx={rows_idx}")
        