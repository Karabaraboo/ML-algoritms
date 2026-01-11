import pandas as pd
import numpy as np
import random

class MyLineReg():    
    def __init__(self, 
                 n_iter=100, 
                 learning_rate=0.1,
                 weights=None, 
                 metric=None,           # Метрики для оценки качества модели. mae, mse, mape, rmse, r2
                 reg=None,              # Выполнение регуляризации весов. l1, l2, elasticnet
                 l1_coef=0,             # Коэффициент L1-регуляризации [0.0, 1.0]
                 l2_coef=0,             # Коэффициент L2-регуляризации [0.0, 1.0]
                 sgd_sample=None,       # Количество образцов в батче при СГС. Если дробное - значит доля от общего числа
                 random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.best_score = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
    
    def fit(self, X_input, y_input, verbose=False):
        # X — все фичи в виде pandas.DataFrame
        # y - целевая переменная, pandas.Series
        # verbose - указывает, на какой итерации выподиться лог
        
        # Преобразование pd.Series в numpy.array (N, 1)
        y = y_input.to_numpy().reshape(-1, 1)    # Сразу, чтобы был столбец и не возникало проблем с np.dot.
        #  Добавление фиктивного столбца из 1 при коэффициенте w0. (N, N_features+1)
        X = np.insert(X_input.to_numpy(), 0, 1, axis=1)

        # Создание исходного вектора весов. Длина = количеству столбцов в X. (N_features+1, 1)
        self.weights = np.ones((X.shape[1], 1))   # Дополнительная размерность, чтобы точно был вектор, и не было проблем np.dot

        random.seed(self.random_state)
        if self.sgd_sample:
            batch_size = int(self.sgd_sample) if self.sgd_sample >= 1 else round(X.shape[0] * self.sgd_sample)
        else:
            batch_size = X.shape[0]
        n = len(y)

        # Цикл итераций
        for i in range(1, self.n_iter + 1):
            # Определение скорости обучения, если она задана динамически
            learning_rate = self.learning_rate(i) if callable(self.learning_rate) else self.learning_rate
            
            # Определение номеров строк X для обучающей выборки
            sample_rows_idx = random.sample(range(X.shape[0]), batch_size)

            # Выборка в батче для SGD
            X_batch = X[sample_rows_idx, :]
            y_batch = y[sample_rows_idx, :]

            # Вектор предсказаний
            y_pred = np.matmul(X, self.weights)    # (N,1)
            y_pred_batch = y_pred[sample_rows_idx, :]    # (N,1)
            
            # Вычисление регуляризации
            reg, grad_reg = (0, 0)    # По умолчанию, если self.reg = None
            if self.reg:
                reg_function = getattr(self, f"reg_{self.reg}", None)
                grad_reg_function = getattr(self, f"grad_reg_{self.reg}", None)
                reg = reg_function(self.weights, [self.l1_coef, self.l2_coef])
                grad_reg = grad_reg_function(self.weights, [self.l1_coef, self.l2_coef])

            # Вычисление loss-функции
            MSE = 1 / n * np.sum((y_pred - y)**2)
            loss = MSE + reg

            # Вычисление градиента loss-функции
            grad_MSE = 2 / batch_size * np.matmul((y_pred_batch - y_batch).T, X_batch)  # (1, N_features+1)
            grad = grad_MSE.T + grad_reg

            # Обновлённые веса
            self.weights -= learning_rate * grad

            if verbose:
                if self.metric:
                    log_metric = f" | loss: {loss} | {self.metric}: {getattr(self, self.metric, None)(y_pred, y, n)}"
                else: 
                    log_metric = f" | loss: {loss}"
                
                if callable(self.learning_rate):
                    log_learning_rate = f" | learning_rate: {learning_rate}"
                else:
                    log_learning_rate = ""
                
                if i == 1:
                    print("start", log_metric, log_learning_rate)
                elif i % verbose == 0:
                    print(i, log_metric, log_learning_rate)
        
        if self.metric:
            y_pred = np.matmul(X, self.weights)
            self.best_score = getattr(self, self.metric, None)(y_pred, y, n)
            

    def get_coef(self):
        return self.weights[1:]

    def predict(self, X_input):
        # X - матрица фичей, pd.DataFrame

        X = np.insert(arr=X_input.to_numpy(), obj=0, values=1, axis=1)
        y_pred = np.matmul(X, self.weights)

        return y_pred
    
    def get_best_score(self):
        return self.best_score

    # Метрики качества модели
    '''
    Использовал @staticmethod, чтобы можно было вынести перечень метрик metrics_map в сам класс.
    Методы не используют self, поэтому они статичные. Это было, кажется, наиболее простым и правильным решением'''
    @staticmethod
    def mse(y_prediction, y_real, n=None):
        if not n:
            n = len(y_prediction)
        return 1 / n * np.sum((y_prediction - y_real)**2)
    
    @staticmethod
    def mae(y_prediction, y_real, n=None):
        if not n:
            n = len(y_prediction)
        return 1 / n * np.sum(np.abs(y_real - y_prediction))

    @staticmethod
    def rmse(y_prediction, y_real, n=None):
        if not n:
            n = len(y_prediction)
        return np.sqrt(1 / n * np.sum((y_real - y_prediction)**2))
    
    @staticmethod
    def mape(y_prediction, y_real, n=None):
        if not n:
            n = len(y_prediction)
        return 100 / n * np.sum(np.abs((y_real - y_prediction) / y_real))
    
    @staticmethod
    def r2(y_prediction, y_real, n=None):
        if not n:
            n = len(y_prediction)
        return 1 - np.sum((y_real - y_prediction)**2) / np.sum((y_real - y_real.mean())**2)
    
    # Вычисление регуляризации
    @staticmethod
    def reg_l1(weights, coefs):
        reg = coefs[0] * np.sum(np.abs(weights))
        #grad = coefs[0] * np.sign(weights)
        return  reg #(reg, grad)
    
    @staticmethod
    def reg_l2(weights, coefs):
        reg = coefs[1] * np.sum(weights**2)
        #grad = 2 * coefs[1] * weights
        return reg #(reg, grad)
    
    @staticmethod
    def reg_elasticnet(weights, coefs):
        reg = coefs[0] * np.sum(np.abs(weights)) + coefs[1] * np.sum(weights**2)
        #grad = coefs[0] * np.sign(weights) + 2 * coefs[1] * weights
        return reg #(reg, grad)
    
    @staticmethod
    def grad_reg_l1(weights, coefs):
        grad = coefs[0] * np.sign(weights)
        return  grad
    
    @staticmethod
    def grad_reg_l2(weights, coefs):
        grad = 2 * coefs[1] * weights
        return grad
    
    @staticmethod
    def grad_reg_elasticnet(weights, coefs):
        grad = coefs[0] * np.sign(weights) + 2 * coefs[1] * weights
        return grad




