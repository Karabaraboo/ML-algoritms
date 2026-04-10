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
                 oob_score = None,
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
        self.fi = {}                            # словарь с важностью фичей
        self.oob_score = oob_score              # метрика расчёта oob-error. 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
        self.oob_score_ = 0                     # рассчитаное значение oob-error

    def __str__(self):
        discription = []

        for parameter, value in self.__dict__.items():
            discription.append(f"{parameter}={value}")

        return f"MyForestClf class: {', '.join(discription)}"
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        features = X.columns.to_list()
        self.fi = dict.fromkeys(features, 0)

        # Количество фич и сэмплов для построения дерева
        cols_cnt = round(self.max_features * X.shape[1])
        rows_cnt = round(self.max_samples * X.shape[0])

        self.trees = [None] * self.n_estimators             # Список деревьев
        
        if self.oob_score:
            oob_predict = np.zeros(X.shape[0])
            oob_tree_cnt = np.zeros(X.shape[0])

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
            
            # Запись дерева и подсчёт суммарного числа листов
            self.trees[tree_number] = tree
            self.leafs_cnt += tree.leafs_cnt

            # Важность фичей
            for feature in cols_names:
                self.fi[feature] += tree.fi[feature] * rows_cnt / X.shape[0]

            # Расчёт oob-error
            if self.oob_score:
                # Маска для элементов, не вошедших в обучение дерева
                oob_msk = ~np.isin(np.arange(X.shape[0]), rows_idx)

                # На этих элементах проводим предсказание
                oob_predict[oob_msk] += tree.predict_proba(X[oob_msk][cols_names]).to_numpy()
                oob_tree_cnt[oob_msk] += 1

        # Подсчёт oob-error после построения всего леса
        if self.oob_score:
            # усреднённые вероятности предсказаний деревьев
            y_predict_proba = np.divide(oob_predict, oob_tree_cnt, out=np.full_like(oob_predict, np.nan), where=oob_tree_cnt != 0)
            nan_msk = np.isnan(y_predict_proba)
            
            # oob-error
            self.oob_score_ = getattr(self, self.oob_score)(y_predict_proba[~nan_msk], y.to_numpy()[~nan_msk])

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

    # Расчёт метрик
    @staticmethod
    def confusion_matrix(y_predict: np.ndarray, y_true: np.ndarray) -> dict:
        conf_matr = {
            'TP': np.sum((y_predict == 1) & (y_true == 1)),
            'FP': np.sum((y_predict == 1) & (y_true == 0)),
            'FN': np.sum((y_predict == 0) & (y_true == 1)),
            'TN': np.sum((y_predict == 0) & (y_true == 0))
        }
        return conf_matr
    
    @staticmethod
    def accuracy(y_predict_proba: np.ndarray, y_true: np.ndarray) -> float:
        # accuracy = (TP + TN) / (TP + FP + TN + FN)
        y_predict = np.where(y_predict_proba > 0.5, 1, 0)
        conf_matrix = MyForestClf.confusion_matrix(y_predict, y_true)
        return (conf_matrix['TP'] + conf_matrix['TN']) / (conf_matrix['TP'] + conf_matrix['FP'] + conf_matrix['TN'] + conf_matrix['FN'])
    
    @staticmethod
    def precision(y_predict_proba: np.ndarray, y_true: np.ndarray) -> float:
        # precision = TP / (TP + FP)
        y_predict = np.where(y_predict_proba > 0.5, 1, 0)
        conf_matrix = MyForestClf.confusion_matrix(y_predict, y_true)
        return conf_matrix['TP'] / (conf_matrix['TP'] + conf_matrix['FP'])
    
    @staticmethod
    def recall(y_predict_proba: np.ndarray, y_true: np.ndarray) -> float:
        # recall = TP / (TP + FN)
        y_predict = np.where(y_predict_proba > 0.5, 1, 0)
        conf_matrix = MyForestClf.confusion_matrix(y_predict, y_true)
        return conf_matrix['TP'] / (conf_matrix['TP'] + conf_matrix['FN'])
    
    @staticmethod
    def f1(y_predict_proba: np.ndarray, y_true: np.ndarray) -> float:
        # f1 = 2 * precison * recall / (precision + recall)
        y_predict = np.where(y_predict_proba > 0.5, 1, 0)
        
        precision = MyForestClf.precision(y_predict, y_true)
        recall = MyForestClf.recall(y_predict, y_true)
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def roc_auc(y_predict_proba: np.ndarray, y_true: np.ndarray):
        '''
        По оси X - FPR (false positive rate) = FP / (FP + TN) - сколько отнесено к положительным среди всех отрицетельных
        По оси Y - TPR (true positive rate) = recall = TP / (TP + FN) - сколько отнесено к положительным среди всех положительных
        '''
        # roc_auc = 1 / (P * N) * Sum_i * Sum_j (I[y_i < y_j] * I[a_i < a_j])
        # y - класс
        # a - значение функции вероятности
        y_predict_proba = y_predict_proba.round(decimals=10)
        #y_true = y_true.round(decimals=10)

        P = np.sum(y_true)
        N = y_true.size - P

        sorted_indices = np.argsort(-y_predict_proba)      # По убыванию
        
        # Расчёт сумм в формуле AUC
        ones_before = 0
        ones_group = 0
        zeros_group = 0
        sum_total = 0
        i = 0
        while True:
            if i >= len(sorted_indices):
                break
            else:
                current = sorted_indices[i]
                if i == len(sorted_indices) - 1:
                    next = None
                else:
                    # current = sorted_indices[i]
                    next = sorted_indices[i + 1]
            
            if next and y_predict_proba[current] == y_predict_proba[next]:       # В группе с одним скором
                if y_true[current] == 0:                                # Запоминаем текущее число
                    zeros_group += 1
                else:
                    ones_group += 1
                while y_predict_proba[current] == y_predict_proba[next]:    # Проходим по группе, меняя next
                    if y_true[next] == 0:                                   # Считаем нули и единицы
                        zeros_group += 1
                    else:
                        ones_group += 1
                    i += 1
                    if i >= len(sorted_indices) - 1:
                        break
                    else:
                        next = sorted_indices[i + 1]
                
                sum_total += (ones_before + 0.5 * ones_group) * zeros_group
                ones_before += ones_group
                ones_group = 0
                zeros_group = 0
                i += 1
            else:
                if y_true[current] == 0:
                    sum_total += ones_before
                else:
                    ones_before += 1
                i += 1

        return sum_total / (P * N)

