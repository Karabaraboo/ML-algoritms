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
                 max_depth=5,                   # максимальная глубина (без учёта листьев)
                 min_samples_split=2,           # минимальное количество объектов в узле, когда ещё можно разбить
                 max_leafs=20,                  # максимальное количество листьев
                 verbose=False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 0
        self.verbose = verbose

    def __str__(self):
        description = []

        for name, value in self.__dict__.items():
            description.append(f"{name} = {value}")

        return f"MyTreeReg class: {', '.join(description)}"
    
    def get_best_split(self, X: pd.DataFrame, y: pd.Series, features: np.ndarray=None) -> tuple:
        if isinstance(X, pd.DataFrame):
            features = X.columns.tolist()
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        
        # Исходные значения
        best_feature = None
        best_gain = -1
        best_split = None

        for column_idx in range(len(features)):
            # MSE в текущем узле
            MSE_node = self.mse(y)

            # Составление списка разделителей
            sorted_values = np.unique(X[:, column_idx])         # Возвращает уникальные значения и уже отсортированные
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
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        features = X.columns.tolist()
        X = X.to_numpy()
        y = y.to_numpy()

        self.leafs_cnt = 0      # На случай повторного вызова fit

        self.tree = self.build_tree(X, y, features, np.arange(X.shape[0]), 1)

    def build_tree(self, X: pd.DataFrame, y: pd.Series, features: list, indices: np.ndarray, depth: int):
        # Условие на лист
        condition_1 = depth > self.max_depth                                # по глубине (без листьев)
        condition_2 = y.size < max(2, self.min_samples_split)               # по числу элементов в узле, но не меньше 1
        condition_3 = self.leafs_cnt + 1 >= self.max_leafs and depth > 1    # по числу листьев
        
        
        if condition_1 or condition_2 or condition_3:
            if self.verbose:  
                print("*********\nУсловия на лист")  
                print(f"Условие по глубине: {condition_1}, Условие по числу элементов: {condition_2}, Условие по числу листов: {condition_3}")
            
            # Если условие выполняется, то это лист
            self.leafs_cnt += 1

            return Node(value=y[indices].mean())
        
        best_split = self.get_best_split(X, y, features)

        left_msk = X[indices, features.index(best_split[0])] <= best_split[1]
        if self.verbose:
            print('**************\nРазбиение узла')
            print(f"фича: {best_split[0]}, разделитель {best_split[1]}")
            print(f"left_msk = {left_msk}")

        tree_left = self.build_tree(X, y, features, indices[left_msk], depth + 1)
        tree_right = self.build_tree(X, y, features, indices[~left_msk], depth + 1)
        
        return Node(feature=best_split[0],
                    threshold=best_split[1],
                    tree_left=tree_left,
                    tree_right=tree_right)


    @staticmethod
    def mse(y: np.ndarray) -> int:
        return ((y - y.mean())**2).mean()