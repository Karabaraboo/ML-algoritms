import numpy as np
import pandas as pd
import random

class MyLogReg():
    # Логистическая регрессия
    def __init__(self,
                 n_iter=10,
                 learning_rate=0.1,
                 metric=None,
                 reg=None,              # Выполнение регуляризации весов. l1, l2, elasticnet
                 l1_coef=0,             # Коэффициент L1-регуляризации [0.0, 1.0]
                 l2_coef=0,             # Коэффициент L2-регуляризации [0.0, 1.0]
                 sgd_sample=None,       # Количество образцов в батче при СГС. Если дробное - значит доля от общего числа
                 random_state=42):   
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None     # Хранение весов модели
        self.metric = metric    # accuracy, precision, recall, f1, roc_auc
        self.best_score = None
        self.reg = reg
        self.reg_l1 = l1_coef
        self.reg_l2 = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state
    
    def __str__(self):
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X_df: pd.DataFrame, y_ser: pd.Series, verbose: bool=False):
        X = np.column_stack((np.ones(X_df.shape[0]), X_df.to_numpy()))  # (N, N_feat+1)
        y = y_ser.to_numpy()[:, np.newaxis]                             # (N, 1)
        self.weights = np.ones((X.shape[1], 1))                         # (N_feat+1, 1)

        n = X.shape[0]

        random.seed(self.random_state)
        if self.sgd_sample:
            batch_size = int(self.sgd_sample) if self.sgd_sample >= 1 else round(X.shape[0] * self.sgd_sample)
        else:
            batch_size = n

        for i in range(1, self.n_iter + 1):
            # Строки батча
            sample_rows_idx = random.sample(range(X.shape[0]), batch_size)

            # Выборка в батче
            X_batch = X[sample_rows_idx, :]   # Можно просто X[sample_rows_idx]
            y_batch = y[sample_rows_idx, :]

            # Предсказание модели
            y_pred = 1 / (1 + np.exp(-np.matmul(X, self.weights)))      # (N, 1)
            y_pred_batch = y_pred[sample_rows_idx]

            # Loss-функция
            # Учёт регуляризации
            regul, grad_regul = (0, 0)
            if self.reg:
                regul_function = getattr(self, self.reg)
                regul, grad_regul = regul_function(self.weights, [self.reg_l1, self.reg_l2])

            eps = 1e-15   # Чтобы избежать +-inf в логарифме
            logLoss = -1 / n * np.sum(y * np.log(y_pred + eps) + (1 - y) * np.log(y_pred + eps)) + regul

            # Градиент loss-функции
            grad = 1 / n * np.matmul((y_pred_batch - y_batch).T, X_batch).T + grad_regul     # (N_feat+1, 1)

            # Обновление весов
            # Определение скорости обучения, если она задана динамически
            learning_rate = self.learning_rate(i) if callable(self.learning_rate) else self.learning_rate
            self.weights -= learning_rate * grad

            # Вывод логов
            if verbose:
                if i % verbose == 0 or i == 1:
                    if i == 1:
                        log_start = "start "
                    else:
                        log_start = i

                    log_loss = f"| loss: {logLoss} "
                    
                    log_metric = ""
                    if self.metric:
                        metric_function = getattr(self, self.metric)
                        if self.metric == "roc_auc":
                            metric = metric_function(self.predict_proba(X_df), y_ser)
                        else:
                            metric = metric_function(self.predict(X_df), y_ser)
                        log_metric = f"| {self.metric}: {metric} "

                    if callable(self.learning_rate):
                        log_learning_rate = f"| learning_rate: {learning_rate}"
                    else:
                        log_learning_rate = ""

                    print(log_start, log_loss, log_metric, log_learning_rate)
        
        if self.metric:
            metric_function = getattr(self, self.metric)
            if self.metric == "roc_auc":
                self.best_score = metric_function(self.predict_proba(X_df), y_ser)
            else:
                self.best_score = metric_function(self.predict(X_df), y_ser)

    def get_coef(self):
        return self.weights[1:]

    def predict_proba(self, X_df: pd.DataFrame):
        X = np.insert(X_df.to_numpy(), 0, 1, axis=1)
        y_pred = 1 / (1 + np.exp(-np.matmul(X, self.weights)))
        return pd.Series(y_pred[:, 0])
    
    def predict(self, X_df: pd.DataFrame):
        predict_proba = self.predict_proba(X_df)
        predict_proba[predict_proba > 0.5] = 1
        predict_proba[predict_proba <= 0.5] = 0
        return predict_proba.astype(int)
    
    def get_best_score(self):
        return self.best_score
    
    # Вычисление метрик
    @staticmethod
    def confusion_matrix(y_predict: pd.Series, y_true: pd.Series):
        y_predict = y_predict.to_numpy()
        y_true = y_true.to_numpy()
        conf_matr = {
            'TP': np.sum((y_predict == 1) & (y_true == 1)),
            'FP': np.sum((y_predict == 1) & (y_true == 0)),
            'FN': np.sum((y_predict == 0) & (y_true == 1)),
            'TN': np.sum((y_predict == 0) & (y_true == 0))
        }
        return conf_matr
    
    @staticmethod
    def accuracy(y_predict: pd.Series, y_true: pd.Series):
        # accuracy = (TP + TN) / (TP + FP + TN + FN)
        conf_matrix = MyLogReg.confusion_matrix(y_predict, y_true)
        return (conf_matrix['TP'] + conf_matrix['TN']) / (conf_matrix['TP'] + conf_matrix['FP'] + conf_matrix['TN'] + conf_matrix['FN'])
    
    @staticmethod
    def precision(y_predict: pd.Series, y_true: pd.Series):
        # precision = TP / (TP + FP)
        conf_matrix = MyLogReg.confusion_matrix(y_predict, y_true)
        return conf_matrix['TP'] / (conf_matrix['TP'] + conf_matrix['FP'])
    
    @staticmethod
    def recall(y_predict: pd.Series, y_true: pd.Series):
        # recall = TP / (TP + FN)
        conf_matrix = MyLogReg.confusion_matrix(y_predict, y_true)
        return conf_matrix['TP'] / (conf_matrix['TP'] + conf_matrix['FN'])
    
    @staticmethod
    def f1(y_predict: pd.Series, y_true: pd.Series):
        # f1 = 2 * precison * recall / (precision + recall)
        precision = MyLogReg.precision(y_predict, y_true)
        recall = MyLogReg.recall(y_predict, y_true)
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def roc_auc(y_predict_proba: pd.Series, y_true: pd.Series):
        # roc_auc = 1 / (P * N) * Sum_i * Sum_j (I[y_i < y_j] * I[a_i < a_j])
        # y - класс
        # a - значение функции вероятности
        # y_predict = y_predict.to_numpy()
        y_predict_proba = y_predict_proba.to_numpy()
        y_true = y_true.to_numpy().round(decimals=10)

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
                    # if y_true[current] == 0:
                    #     sum_total += ones_before
                    # break
                    next = None
                else:
                    current = sorted_indices[i]
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
    
    # Линеаризация
    @staticmethod
    def l1(weights: np.ndarray, coef: list):
        regul = coef[0] * np.sum(np.abs(weights))
        grad = coef[0] * np.sign(weights)
        return (regul, grad)
    
    @staticmethod
    def l2(weights: np.ndarray, coef: list):
        regul = coef[1] * np.sum(weights**2)
        grad = 2 * coef[1] * weights
        return (regul, grad)
    
    @staticmethod
    def elasticnet(weights: np.ndarray, coef: list):
        regul = coef[0] * np.sum(np.abs(weights)) + coef[1] * np.sum(weights**2)
        grad = coef[0] * np.sign(weights) + 2 * coef[1] * weights
        return (regul, grad)
    
