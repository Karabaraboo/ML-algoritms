import numpy as np
import pandas as pd

class MyKNNReg():
    def __init__(self, k=3):
        self.k = k              # количество ближайших соседей
        self.train_size = (0, 0)    # размер обучающей выборки

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
    
