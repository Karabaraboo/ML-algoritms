import numpy as np
import pandas as pd

class MyKNNClf():
    def __init__(self, 
                 k=3,                   # Число ближайших соседей 
                 metric='euclidean',    # Метрика расчёта расстояния. ['euclidean', 'chebyshev', 'manhattan', 'cosine']
                 weight='uniform'):     # Взвешенный kNN. ['uniform', 'rank', 'distance']
        self.k = k          # Количество ближайших соседей
        self.train_size = (0, 0)
        self.metric = metric
        self.weight = weight
    
    def __str__(self):
        description = []
        for param_name, param_value in self.__dict__.items():
            description.append(f"{param_name}={param_value}")
        return f"MyKNNClf class: {', '.join(description)}"  # Можно использовать атрибут __dict__ экземпляра
    
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
        distances = X.apply(distance_func, axis=1).to_numpy()              # (N_pred, N)
        # Индексы k ближайших соседей
        neighbours_indices = np.argsort(distances)[:, :self.k]
        neighbours_classes = self.y_train.to_numpy()[neighbours_indices]
        # Создание массива с номерами строк для корректного использования broadcasting при обращении по индексам столбцов далее
        rows = np.arange(distances.shape[0])[:, np.newaxis]
        neighbours_distances = distances[rows, neighbours_indices] # Тут rows дополняется столбцами до количества столбов indices
        #print('indices in predict_proba:', neighbours_indices, sep='\n')
        if self.weight == 'uniform':
            y_predict_proba = np.sum(neighbours_classes, axis=1) / self.k

        elif self.weight == 'rank': 
            y_predict_proba = np.sum(neighbours_classes / np.arange(1, self.k + 1), axis=1) / np.sum(1 / np.arange(1, self.k + 1))

        elif self.weight == 'distance':
            # Массив номеров строк, т.к. neighbours_distances двумерный
            y_predict_proba = np.sum(neighbours_classes / neighbours_distances, axis=1) / np.sum(1 / neighbours_distances, axis=1)

        else:
            raise ValueError(f"Invalid weight: {self.weight}")
        return pd.Series(y_predict_proba)

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


