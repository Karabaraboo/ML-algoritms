import numpy as np
import pandas as pd

class MyLogReg():
    # Логистическая регрессия
    def __init__(self,
                 n_iter=10,
                 learning_rate=0.1,
                 metric=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None     # Хранение весов модели
        self.metric = metric    # accuracy, precision, recall, f1, roc_auc
    
    def __str__(self):
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X_df: pd.DataFrame, y_ser: pd.Series, verbose: bool=False):
        X = np.column_stack((np.ones(X_df.shape[0]), X_df.to_numpy()))  # (N, N_feat+1)
        y = y_ser.to_numpy()[:, np.newaxis]                             # (N, 1)
        self.weights = np.ones((X.shape[1], 1))                         # (N_feat+1, 1)

        n = X.shape[0]
        for i in range(1, self.n_iter + 1):
            # Предсказание модели
            y_pred = 1 / (1 + np.exp(-np.matmul(X, self.weights)))      # (N, 1)

            # Loss-функция
            eps = 1e-15   # Чтобы избежать +-inf в логарифме
            logLoss = -1 / n * np.sum(y * np.log(y_pred + eps) + (1 - y) * np.log(y_pred + eps))

            # Градиент loss-функции
            grad = 1 / n * np.matmul((y_pred - y).T, X)                 # (1, N_feat+1)

            # Обновление весов
            self.weights -= self.learning_rate * grad.T

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
                        if self.metric == "roc auc":
                            metric = metric_function(self.predict_proba(X_df), y_ser)
                        else:
                            metric = metric_function(self.predict(X_df), y_ser)
                        log_metric = f"| {self.metric}: {metric} "

                    print(log_start, log_loss, log_metric)

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
        recall = MyLogReg.precision(y_predict, y_true)
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
        ones_before = 0             # Количество единиц перед текущей i-позицией и выше скором. yi < yj, ai < aj
        ones_score = y_true[0]      # Количество единиц перед текущей позицией с таким же скором. yi < yj, ai = aj
        zeros_score = 1 - y_true[0]
        sum_total = 0               # Общая сумма, которая складывается из единиц (yi < yj, ai < aj) и 0,5*единиц (yi < yj, ai = aj)
        # score = 2                 # Скор на предыдущей позиции. 2 - чтобы начать итерацию.

        for position in range(1, len(sorted_indices)):
            current = sorted_indices[position]   # Индекс в исходных массивах
            previous = sorted_indices[position - 1]
            
            if y_predict_proba[current] == y_predict_proba[previous]:   # В группе одного скора
                if y_true[current] == 0:
                    zeros_score += 1
                else:
                    ones_score += 1
            else:
                if ones_score * zeros_score:    # Если вышли из группы одного скора, то zeros_score и ones_score != 0
                    sum_total += (ones_before + 0.5 * ones_score ) * zeros_score
                ones_before += ones_score
                if y_true[current] == 0:
                    sum_total += ones_before
                    zeros_score = 1
                    ones_score = 0
                else:
                    ones_score = 1
                    zeros_score = 0

        return sum_total / (P * N)
