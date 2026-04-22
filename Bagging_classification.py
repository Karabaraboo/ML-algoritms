import numpy as np
import pandas as pd
import random
import copy

class MyBaggingClf():
    def __init__(self,
                 estimator=None,
                 n_estimators: int=10,
                 max_samples: float=1.0,
                 oob_score: str=None,
                 random_state: int=42):
        self.estimator = estimator              # экземпляр базового класса
        self.n_estimators = n_estimators
        self.max_samples = max_samples          # доля примеров для обучайщей модели. [0.0, 1.0]
        self.oob_score = oob_score              # метрика. accuracy, precision, recall, f1, roc_auc
        self.oob_score_ = 0                     # значение oob-error
        self.random_state = random_state

    def __str__(self):
        description = []

        for name, value in self.__dict__.items():
            description.append(f"{name}={value}")

        return f"MyBaggingClf class: {', '.join(description)}"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Список для записи обученных моделей
        self.estimators = []
        
        # Количество сэмплов для обучения в модели
        rows_cnt = round(X.shape[0] * self.max_samples)

        random.seed(self.random_state)

        # Номера строк для обучения
        rows_samples = [random.choices(np.arange(X.shape[0]), k=rows_cnt) for _ in range(self.n_estimators)]

        if self.oob_score:
            y_predict_proba_oob = np.zeros(X.shape[0])
            predict_proba_cnt_oob = np.zeros(X.shape[0])

        for estimator_number in range(self.n_estimators):
            # Модель для обучения
            estimator = copy.deepcopy(self.estimator)

            # Строки матрицы фичей для обучения
            rows_sample_num = rows_samples[estimator_number]
            # Обучение
            estimator.fit(X.iloc[rows_sample_num], y.iloc[rows_sample_num])

            if self.oob_score:
                oob_msk = ~np.isin(np.arange(X.shape[0]), rows_sample_num)

                y_predict_proba_oob[oob_msk] += estimator.predict_proba(X[oob_msk]).to_numpy()
                predict_proba_cnt_oob[oob_msk] += 1

            # Запись экземпляра
            self.estimators.append(estimator)

        if self.oob_score:
            y_predict_proba_oob = np.divide(y_predict_proba_oob, 
                                            predict_proba_cnt_oob, 
                                            out=np.full_like(y_predict_proba_oob, np.nan), 
                                            where=predict_proba_cnt_oob != 0)
            oob_score_msk = ~np.isnan(y_predict_proba_oob)

            self.oob_score_ = getattr(self, self.oob_score)(y_predict_proba_oob[oob_score_msk], y.to_numpy()[oob_score_msk])

    def predict(self, X: pd.DataFrame, type: str) -> pd.Series:
        if type == 'mean':
            return (self.predict_proba(X) > 0.5).astype(int)
        elif type == 'vote':
            return self.predict_label(X)
        else:
            return f"Undefined type"

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        '''Метод возвращает предсказания в виде вероятностей'''
        X = X.reset_index(drop=True)
        prediction = np.zeros(X.shape[0])
        for estimator in self.estimators:
            prediction += estimator.predict_proba(X)

        return pd.Series(prediction / self.n_estimators, index=X.index)
    
    def predict_label(self, X: pd.DataFrame) -> pd.Series:
        '''Метод возвращает предсказания в виде классов'''
        X = X.reset_index(drop=True)
        prediction = np.zeros(X.shape[0])

        for estimator in self.estimators:
            prediction += estimator.predict(X) / self.n_estimators

        return pd.Series(np.where(prediction > 0.5, 1, 0), index=X.index)
    
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
        conf_matrix = MyBaggingClf.confusion_matrix(y_predict, y_true)
        return (conf_matrix['TP'] + conf_matrix['TN']) / (conf_matrix['TP'] + conf_matrix['FP'] + conf_matrix['TN'] + conf_matrix['FN'])
    
    @staticmethod
    def precision(y_predict_proba: np.ndarray, y_true: np.ndarray) -> float:
        # precision = TP / (TP + FP)
        y_predict = np.where(y_predict_proba > 0.5, 1, 0)
        conf_matrix = MyBaggingClf.confusion_matrix(y_predict, y_true)
        return conf_matrix['TP'] / (conf_matrix['TP'] + conf_matrix['FP'])
    
    @staticmethod
    def recall(y_predict_proba: np.ndarray, y_true: np.ndarray) -> float:
        # recall = TP / (TP + FN)
        y_predict = np.where(y_predict_proba > 0.5, 1, 0)
        conf_matrix = MyBaggingClf.confusion_matrix(y_predict, y_true)
        return conf_matrix['TP'] / (conf_matrix['TP'] + conf_matrix['FN'])
    
    @staticmethod
    def f1(y_predict_proba: np.ndarray, y_true: np.ndarray) -> float:
        # f1 = 2 * precison * recall / (precision + recall)
        y_predict = np.where(y_predict_proba > 0.5, 1, 0)
        
        precision = MyBaggingClf.precision(y_predict, y_true)
        recall = MyBaggingClf.recall(y_predict, y_true)
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