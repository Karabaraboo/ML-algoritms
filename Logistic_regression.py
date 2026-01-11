import numpy as np
import pandas as pd

class MyLogReg():
    # Логистическая регрессия
    def __init__(self,
                 n_iter=10,
                 learning_rate=0.1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None     # Хранение весов модели
    
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
                if i == 1:
                    print(f"start | loss: {logLoss}")
                elif i % verbose == 0:
                    print(f"{i} | loss: {logLoss}")

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