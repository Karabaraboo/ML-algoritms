import numpy as np
import pandas as pd

class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature     # Параметр, по которому выполняется разбиение
        self.threshold = threshold   # Порог разбиения
        self.left = None        # Левое поддерево
        self.right = None       # Правое поддерево

        self.value = value      # вероятность первого класса, если лист


class MyTreeClf():
    def __init__(self,
                 max_depth=5,               
                 min_samples_split=2,
                 max_leafs=20):
        self.max_depth = max_depth          # максимальная глубина дерева
        self.min_samples_split = min_samples_split   # мин допустимое кол-во объектов в листе для разбиения
        self.max_leafs = max_leafs          # макс разрешённое кол-во листов в дереве
        self.leafs_cnt = 0                  # Текущее количество листьев в дереве

    
    def __str__(self):
        description = []
        for name, value in self.__dict__.items():
            description.append(f"{name}={value}")
        
        return f"MyTreeClf class: {', '.join(description)}"

    def get_best_split(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """ 
        X - матрица фичей                                             # (N, N_feat)
        y - таргет                                                    # (N,)
        Возвращает:
        col_name - название фичи для разделения
        split_value - значение, по которому будет производиться разделение
        ig - прирост информации
        """
        # X = X.to_numpy()                                                # (N, N_feat)
        # y = y.to_numpy()                                                # (N,)
        self.ig = 0
        self.col_name = ''
        self.split_value = 0

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        # Исходная энтропия
        S0 = self.enthropy(y)

        print(f"S0 = {S0}")
        for column in X:
            sorted_values = np.sort(X[column].unique())
            separators = (sorted_values[:-1] + sorted_values[1:]) / 2

            for sep in separators:
                print(f"Слева {(X[column] <= sep).sum()} элементов, из них {X[column].loc[(X[column] <= sep) & (y == 0)].count()} нулевого класса")
                left_idx = X[column] <= sep     # bool-индексы элементов слева
                right_idx = X[column] > sep     # bool-индексы элементов справа
                left_num = left_idx.sum()       # количество элементов слева
                right_num = right_idx.sum()     # и справа
                S_left = self.enthropy(y[left_idx])
                S_right = self.enthropy(y[right_idx])
                
                print(f"S1 = {S_left}, S2 = {S_right}")
                # Прирост информации
                ig_current = S0 - left_num / (left_num + right_num) * S_left - right_num / (left_num + right_num) * S_right

                print(f"current column is {column}, current sep = {sep}, current ig = {ig_current}")
                if ig_current > self.ig:
                    self.ig = ig_current
                    self.col_name = column
                    self.split_value = sep
                
                print(f"best column is {self.col_name}, best ig = {self.ig}")
        
        return (self.col_name, self.split_value, self.ig)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = X.to_numpy()
        y = y.to_numpy()
        self.tree(X, y, 1)

    def tree(self, X: np.ndarray, y: np.ndarray, depth: int):
        # Если это лист
        if (X.shape[0] < 2 or                        # Если выборка содержит 1 элемент
            y.sum() == y.shape[0] or y.sum() == 0 or        # или в ней один класс
            depth >= self.max_depth or                      # превышена допустимая глубина дерева
            y.shape[0] < self.min_samples_split or          # число элементов меньше минимально допустимого
            self.leafs_cnt > self.max_leafs or              # текущее число листьев превышает заданное максимальное
            (2**depth > self.max_leafs and depth > 1)):     # потенциальное число листьев превышает максимальное (кроме корня дерева)
            # тогда возращаем вероятность первого класса
            self.leafs_cnt += 1
            return Node(value = y.sum() / y.shape[0])
        
        # Если это узел, то разбиваем        
        best_split = self.get_best_split(X, y)
        left_idx = X[best_split[0]] <= best_split[1]
        right_idx = X[best_split[0]] > best_split[1]

        # и вызываем построение 
        left_tree = self.tree(X[left_idx], y[left_idx], depth + 1)
        right_tree = self.tree(X[right_idx], y[right_idx], depth + 1)
        
        # Запись в Node
        return Node(feature=best_split[0],
                    threshold=best_split[1],
                    left=left_tree,
                    right=right_tree)

    @staticmethod
    def enthropy(y):
        p1 = y.sum() / y.shape[0]
        p0 = 1 - p1
        return 0 if p0 * p1 == 0 else -p0 * np.log2(p0) - p1 * np.log2(p1)
    
