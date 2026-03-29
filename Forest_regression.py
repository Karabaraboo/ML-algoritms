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
                 oob_score=None,
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
        self.fi = {}                                # Важность фичей
        self.oob_score = oob_score                  # OOB-error (mae, mse, rmse, mape, r2)
        self.oob_score_ = 0                         # Переменная, хранящая oob_error

        self.verbose = verbose

    def __str__(self):
        parameters = []

        for name, value in self.__dict__.items():
            parameters.append(f"{name}={value}")

        return f"MyForestReg class: {', '.join(parameters)}"
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        features = X.columns.to_list()
        self.fi = dict.fromkeys(features, 0)     # На случай повторного вызова fit
        self.oob_score_ = 0
        
        # Количество фичей и элементов для отбора
        cols_cnt = round(self.max_features * X.shape[1])
        rows_cnt = round(self.max_samples * X.shape[0])

        self.trees = [None] * self.n_estimators      # Список для деревьев
        random.seed(self.random_state)

        for tree_number in range(self.n_estimators):
            cols_name = random.sample(X.columns.to_list(), cols_cnt)
            rows_idx = random.sample(range(X.shape[0]), rows_cnt)

            # Обучение дерева
            tree = MyTreeReg(max_depth=self.max_depth,
                             min_samples_split=self.min_samples_split,
                             max_leafs=self.max_leafs,
                             bins=self.bins,
                             verbose=False)
            tree.fit(X.loc[X.index[rows_idx], cols_name], y.loc[y.index[rows_idx]])
            #tree.fit(X.iloc[rows_idx][cols_name], y.iloc[rows_idx])

            self.leafs_cnt += tree.leafs_cnt            
            self.trees[tree_number] = tree
            
            # Расчёт важности фичей (feature importance)
            for feature in cols_name:
                self.fi[feature] += tree.fi[feature] * rows_cnt / X.shape[0]

            # Расчёт OOB-error
            if self.oob_score:
                if tree_number == 0:
                    y_predict = np.zeros(X.shape[0])             # Массив для суммирования предсказаний
                    trees_oob = np.ones(X.shape[0])                         # Количество деревьев out-of-bag в каждой строке

                oob_msk = np.isin(np.arange(X.shape[0]), rows_idx)          # Индексы строк, не вошедшие в дерево
                y_predict[oob_msk] += tree.predict(X[oob_msk][cols_name]).to_numpy()      # Предсказание на оставшихся строках. X.drop(X.index[rows_idx])
                trees_oob[oob_msk] += 1            # Увеличение числа деревьев в строках, которые oob

            if self.verbose:
                print(f"cols_cnt={cols_cnt}, rows_cnt={rows_cnt},\ncols_idx={cols_name},\nrows_idx={rows_idx}")
                tree.print_tree()
                if self.oob_score:
                    print('tree_predict:\n', tree.predict(X[oob_msk][cols_name]))
                    print(f"oob_idx:\n{oob_msk},\ny_predict:\n{y_predict},\ntrees_oob:\n{trees_oob}")

        if self.oob_score:
            y_predict = np.where(trees_oob != 0, y_predict / trees_oob, np.nan) # Осреднение предсказаний по количеству деревьев
            
            # Расчёт метрики oob_score
            oob_score_idx = ~np.isnan(y_predict)        # Индексы элементов, которые не nan
            self.oob_score_ = getattr(self, self.oob_score)(y_predict[oob_score_idx], y.iloc[oob_score_idx])    # y.drop(y.index[rows_idx])
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        prediction = np.zeros(X.shape[0])

        for tree in self.trees:
            tree_prediction = tree.predict(X)
            prediction += tree_prediction

            if self.verbose:
                print(f"Предсказание дерева: {tree_prediction}")

        return prediction / self.n_estimators
    
    # Метрики для подсчёта OOB-error
    @staticmethod
    def mse(y_prediction, y_real, n=None) -> float:
        if isinstance(y_prediction, pd.Series):
            y_prediction = y_prediction.to_numpy()
        if isinstance(y_real, pd.Series):
            y_real = y_real.to_numpy()
        if not n:
            n = len(y_prediction)
        return 1 / n * np.sum((y_prediction - y_real)**2)
    
    @staticmethod
    def mae(y_prediction, y_real, n=None) -> float:
        if isinstance(y_prediction, pd.Series):
            y_prediction = y_prediction.to_numpy()
        if isinstance(y_real, pd.Series):
            y_real = y_real.to_numpy()
        if not n:
            n = len(y_prediction)
        return 1 / n * np.sum(np.abs(y_real - y_prediction))

    @staticmethod
    def rmse(y_prediction, y_real, n=None) -> float:
        if isinstance(y_prediction, pd.Series):
            y_prediction = y_prediction.to_numpy()
        if isinstance(y_real, pd.Series):
            y_real = y_real.to_numpy()
        if not n:
            n = len(y_prediction)
        return np.sqrt(1 / n * np.sum((y_real - y_prediction)**2))
    
    @staticmethod
    def mape(y_prediction, y_real, n=None) -> float:
        if isinstance(y_prediction, pd.Series):
            y_prediction = y_prediction.to_numpy()
        if isinstance(y_real, pd.Series):
            y_real = y_real.to_numpy()
        if not n:
            n = len(y_prediction)
        return 100 / n * np.sum(np.abs((y_real - y_prediction) / y_real))
    
    @staticmethod
    def r2(y_prediction, y_real, n=None) -> float:
        if isinstance(y_prediction, pd.Series):
            y_prediction = y_prediction.to_numpy()
        if isinstance(y_real, pd.Series):
            y_real = y_real.to_numpy()
        if not n:
            n = len(y_prediction)
        return 1 - np.sum((y_real - y_prediction)**2) / np.sum((y_real - y_real.mean())**2)