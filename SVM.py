import pandas as pd
import numpy as np

class MySVM():
    def __init__(self,
                 n_iter: int=10,                # число итераций
                 learning_rate: float=0.001,    # скорость обучения
                 C = 1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights=None                       # веса модели
        self.b = None                           # отступ гиперплоскости
        self.C = C

    def __str__(self):
        return f"MySVM class: n_iter={self.n_iter}, learning_rate={self.learning_rate}, C={self.C}"
    
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        # X - матрица фичей
        # y - целевая переменная. По-умолчанию, передаётся в виде 0 и 1
        # verbose - метка вывода логов на печать

        y.loc[y == 0] = -1

        # Начальный вектор весов
        self.weights = pd.Series(np.ones(X.shape[1]))                           # pd.Series, (N_feat,)
        self.b = 1

        n = X.shape[0]

        for i in range(1, self.n_iter + 1):
            for row in range(X.shape[0]):
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
                loss = self.weights.dot(self.weights) + 1 / n * max(0, 1 - y_i * (X_i.dot(self.weights) + self.b))

                # # Расчёт y_i * (X_i * w + b)
                # hingeLoss_condition = y.mul(X.dot(self.weights) + self.b)       # pd.Series, (N,)

                # grad_Loss = 2 * self.weights                                    # pd.Series, (N,)
            
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

