import numpy as np
import pandas as pd

class Node():
    def __init__(self, feature=None, threshold=None, tree_left=None, tree_right=None, value=None):
        self.feature = feature          # Фича для деления
        self.threshold = threshold      # Порог фичи
        self.tree_left = tree_left      # Дерево слева
        self.tree_right = tree_right    # Дерево справа

        self.value = value              # Значение, если это лист

class MyTreeReg():
    def __init__(self, 
                 max_depth=5,                   # максимальная глубина
                 min_samples_split=2,           # минимальное количество объектов в листе
                 max_leafs=20):                 # максимальное количество листьев
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs

    def __str__(self):
        description = []

        for name, value in self.__dict__.items():
            description.append(f"{name} = {value}")

        return f"MyTreeReg class: {', '.join(description)}"
    
    def get_best_split(self, X: pd.DataFrame, y: pd.Series, features: np.ndarray=None) -> tuple:
        if isinstance(X, pd.DataFrame):
            features = X.columns.values
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        
        # Исходные значения
        best_feature = None
        best_gain = -1
        best_split = None

        for column_idx in range(features.size):
            # MSE в текущем узле
            MSE_node = self.mse(y)

            # Составление списка разделителей
            sorted_values = np.sort(X[:, column_idx])
            separators = (sorted_values[1:] + sorted_values[:-1]) / 2

            for sep in separators:
                left_msk = X[:, column_idx] <= sep
                right_msk = X[:, column_idx] > sep

                # MSE для групп
                MSE_left = self.mse(y[left_msk])
                MSE_right = self.mse(y[right_msk])

                # Расчёт прироста (gain)
                gain = MSE_node - (left_msk.sum() / y.size * MSE_left + right_msk.sum() / y.size * MSE_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = features[column_idx]
                    best_split = sep
        
        return (best_feature, best_split, best_gain)

    @staticmethod
    def mse(y: np.ndarray) -> int:
        return ((y - y.mean())**2).mean()