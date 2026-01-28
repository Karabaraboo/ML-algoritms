import numpy as np
import pandas as pd

class MyKNNClf():
    def __init__(self, k=3, metric='euclidean'):
        self.k = k          # Количество ближайших соседей
        self.train_size = (0, 0)
        self.metric = metric    # 'euclidean', 'chebyshev', 'manhattan', 'cosine'
    
    def __str__(self):
        return f"MyKNNClf class: k={self.k}"  # Можно использовать атрибут __dict__ экземпляра
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # X - матрица фичей                                     # (N, N_feat)
        # y - целевая переменная                                # (N,)

        self.X_train = X
        self.y_train = y
        self.train_size = X.shape

    def predict(self, X: pd.DataFrame):
        # X - матрица фичей для предсказаний                    # (N_pred, N_feat)
        y_predict_proba = self.predict_proba(X)
        return (y_predict_proba >= 0.5) * 1

    def predict_proba(self, X: pd.DataFrame):
        # X - матрица фичей для предсказаний                    # (N_pred, N_feat)
        distance_func = getattr(self, self.metric)
        distances = X.apply(distance_func, axis=1)              # (N_pred, N)
        # Индексы k ближайших соседей
        indexes = distances.apply(lambda row: np.argsort(row)[0:self.k], axis=1)
        print('indices in predict_proba:', indexes, sep='\n')
        y_predict_proba = indexes.apply(lambda row: self.y_train.iloc[row].sum() / row.shape[0], axis=1)
        return y_predict_proba

    def euclidean(self, x2: pd.Series):
        #print('Пришло: x2', x2)
        return np.sqrt(((self.X_train - x2)**2).sum(axis=1))
    
    def chebyshev(self, x2: pd.Series):
        # self.X_train                                      # (N, N_feat)
        # x2                                                # (N_feat,)
        return np.abs(self.X_train - x2).max(axis=1)              # (N,)

    def manhattan(self, x2: pd.Series):
        # self.X_train                                      # (N, N_feat)
        # x2                                                # (N_feat,)
        return np.abs(self.X_train - x2).sum(axis=1)        # (N,)

    def cosine(self, x2: pd.Series):
        # self.X_train                                      # (N, N_feat)
        # x2                                                # (N_feat,)
        numerator = (self.X_train * x2).sum(axis=1)                             # (N,)
        denominator = np.sqrt((self.X_train**2).sum(axis=1) * (x2**2).sum())    # (N,)
        return 1 - numerator / denominator                  # (N,)


