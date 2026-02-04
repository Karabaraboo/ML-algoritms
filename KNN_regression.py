import numpy as np
import pandas as pd

class MyKNNReg():
    def __init__(self, 
                 k=3,                   # Число ближайших соседей 
                 metric='euclidean',    # Метрика расчёта расстояния. ['euclidean', 'chebyshev', 'manhattan', 'cosine']
                 weight='uniform'):     # Взвешенный kNN. ['uniform', 'rank', 'distance']
        self.k = k
        self.train_size = (0, 0)        # размер обучающей выборки
        self.metric = metric
        self.weight = weight

    def __str__(self):
        description = []
        for param_name, param_value in self.__dict__.items():
            description.append(f"{param_name}={param_value}")
        return f"MyKNNReg class: {','.join(description)}"
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # X - матрица фичей                                         (N, N_feat)
        # y - целевая переменная                                    (N,)
        self.x_train = X.to_numpy()
        self.y_train = y.to_numpy()
        self.train_size = X.shape

    def predict(self, X: pd.DataFrame) -> pd.Series:
        x1 = X.to_numpy()                                       # (n1, m)
        x2 = self.x_train                                       # (n2, m)

        distances = getattr(self, self.metric)(x1, x2)          # (n1, n2)
        # print(distances)
        neighbour_idx = np.argsort(distances)[:, :self.k]       # (n1, k)
        # print(neighbour_idx)
        # print(self.y_train[neighbour_idx])
        neighbour_predict = self.y_train[neighbour_idx]        # (n1, k)
        '''Тут работает механизм Fancy indexing в numpy. result[i, j] = y_train[idx[i, j]]'''

        if self.weight == 'uniform':
            return pd.Series(neighbour_predict.mean(axis=1))
        
        elif self.weight == 'rank':
            weights = 1 / np.arange(1, self.k + 1) / (1 / np.arange(1, self.k + 1)).sum()   # (k,)
            return pd.Series((neighbour_predict * weights).sum(axis=1))
        
        elif self.weight == 'distance':
            '''Необходимо добавить дополнительную ось для строк, чтобы broadcasting
            растянул массив строк и массив столбцов между собой, создав матрицу индексов
            размерностью (n1, k).
            Далее механизм fancy indexing обращается по каждому индексу из массива idx
            к массиву distances и создаёт результат размерности (n1, k)'''
            rows_num = np.arange(distances.shape[0])[:, np.newaxis]                 # (n1, 1)
            cols_num = neighbour_idx                                                # (n1, k)
            reverse_dist = 1 /distances[rows_num, cols_num]                         # (n1, k)
            weights = reverse_dist / np.sum(reverse_dist, axis=1)[:, np.newaxis]    # (n1, k)
            return pd.Series((neighbour_predict * weights).sum(axis=1))

        else:
            raise ValueError(f"Invalid weight: {self.weight}") 

    @staticmethod
    def euclidean(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """У x1 размерность (n1, m), у x2 - (n2, m).
         Чтобы попарно вычитать между собой параметры (в m столбцах), но при этом 
         для каждого объекта из x1 (n1 штук) рассчитать расстояние до каждого объекта x2 (n2 штук)
         добавляем x1 новую ось. 
         Тогда у x1 размерность (n1, 1, m). По правилам broadcasting в numpy итоговый массив
         будет иметь размерность (n1, n2, m), что нам и нужно."""

        return np.sqrt(((x1[:, np.newaxis, :] - x2)**2).sum(axis=2))
    
    @staticmethod
    def chebyshev(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return np.abs(x1[:, np.newaxis, :] - x2).max(axis=2)
    
    @staticmethod
    def manhattan(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return np.abs(x1[:, np.newaxis, :] - x2).sum(axis=2)
    
    @staticmethod
    def cosine(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        # x1 (n1, m)
        # x2 (n2, m)

        numerator = (x1[:, np.newaxis, :] * x2).sum(axis=2)                 # (n1, n2)
        denominator = np.sqrt((x1**2).sum(axis=1)[:, np.newaxis] * (x2**2).sum(axis=1))     # (n1, 1) * (1, n2)
        """По правилам broadcasting массиву x2 добавляется одна ось слева, т.е. (1, n2).
        Итоговая размерность становиться (n1, n2)"""

        # print(f"numerator is \n {numerator}, \ndenominator is {denominator}")
        return 1 - numerator / denominator