import pandas as pd
import numpy as np
import random

class MySVM():
    def __init__(self,
                 n_iter: int=10,                # число итераций
                 learning_rate: float=0.001,    # скорость обучения
                 C = 1,                         # Коэффициент учёта мягкого зазора
                 sgd_sample=None,               # Количество образцов в батче при СГС. Если дробное - значит доля от общего числа
                 random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights=None                       # веса модели
        self.b = None                           # отступ гиперплоскости
        self.C = C                              # Коэффициент перед слагаемым в loss, отвечающим за классификацию
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self):
        return f"MySVM class: n_iter={self.n_iter}, learning_rate={self.learning_rate}, C={self.C}, sgd_sample={self.sgd_sample}"
    
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        # X - матрица фичей                                                     # (N, N_feat)
        # y - целевая переменная. По-умолчанию, передаётся в виде 0 и 1         # (N,)
        # verbose - метка вывода логов на печать

        y.loc[y == 0] = -1

        # Начальный вектор весов
        self.weights = pd.Series(np.ones(X.shape[1]))                           # pd.Series, (N_feat,)
        self.b = 1

        n = X.shape[0]

        # Стохастический градиентный спуск. Подготовка данных о батче
        random.seed(self.random_state)
        if self.sgd_sample:
            batch_size = int(self.sgd_sample) if self.sgd_sample > 1 else int(self.sgd_sample * X.shape[0])
        else:
            batch_size = n

        for i in range(1, self.n_iter + 1):

            if self.sgd_sample:
                sample_rows_idx = random.sample(range(n), batch_size)
            else:
                sample_rows_idx = range(n)

            for row in sample_rows_idx:
                X_i = X.iloc[row]                                               # pd.Series, (N_feat,)
                y_i = y.iloc[row]                                               # np.int

                if y_i * (X_i.dot(self.weights) + self.b) >= 1:
                    grad_loss_w = 2 * self.weights
                    grad_loss_b = 0
                else:
                    grad_loss_w = 2 * self.weights - y_i * X_i * self.C
                    grad_loss_b = -y_i * self.C

                # Обновление весов
                self.weights -= self.learning_rate * grad_loss_w
                self.b -= self.learning_rate * grad_loss_b

            # loss = ||w||**2 + 1 / N * sum_i{max(0, 1 - y_i * (x_i * w + b))}
            loss = self.weights.dot(self.weights) + 1 / n * np.sum(np.maximum(0, 1 - y.mul(X.dot(self.weights) + self.b)))
            
            if verbose:
                if i == 1:
                    l_start = 'start'
                else:
                    l_start = i

                l_loss = f"| loss: {loss}"

                if i % verbose == 0:
                    print(l_start, l_loss, sep=" ")

    def get_coef(self):
        return (self.weights, self.b)
    
    def predict(self, X: pd.DataFrame):
        # X  (N, N_feat)
        # w  (N_feat,)
        # y_predict  (N,)

        y_predict = np.sign(X.dot(self.weights) + self.b)        
        return y_predict.replace(-1, 0).astype(int)

