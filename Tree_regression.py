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
                 verbose=False,
                 bins=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 0
        self.verbose = verbose
        self.bins = bins
        self.fi = {}

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
            if self.bins:
                separators = self.thresholds[column_idx]
            else:
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

        # Оказалось, что pd.nunique работает быстрее np.unique.size, т.к не требует сортировки.
        # Но для поиска разделителей всё равно нужны уникальные значения и отсортированные
        if self.bins:
            self.get_separators(X)      # Возвращает self.thresholds - список разделителей для столбцов

        # Создание словаря с важностью фичей. Все значения = 0
        self.fi = dict.fromkeys(features, 0)

        self.leafs_cnt = 0      # На случай повторного вызова fit

        self.tree = self.build_tree(X, y, features, np.arange(X.shape[0]), 1)

    def build_tree(self, X: np.ndarray, y: np.ndarray, features: list, indices: np.ndarray, depth: int, potential: int=2):
        # Условие на лист
        condition_1 = depth > self.max_depth                                # по глубине (без листьев)
        condition_2 = y[indices].size < max(2, self.min_samples_split)               # по числу элементов в узле, но не меньше 1
        condition_3 = self.leafs_cnt + potential > self.max_leafs and depth > 1    # по числу листьев
        
        if condition_1 or condition_2 or condition_3:
            if self.verbose:  
                print("*********\nУсловия на лист")  
                print(f"Условие по глубине: {condition_1},\nУсловие по числу элементов: {condition_2},\nУсловие по числу листов: {condition_3}")
                print(f"{self.leafs_cnt + 1}-ый лист, значение {y[indices].mean()}")
            
            # Если условие выполняется, то это лист
            self.leafs_cnt += 1

            return Node(value=y[indices].mean())
        
        best_split = self.get_best_split(X[indices], y[indices], features)

        left_msk = X[indices, features.index(best_split[0])] <= best_split[1]
        condition_4 = len(indices[left_msk]) == 0 or len(indices[~left_msk]) == 0

        if condition_4:
            '''Это лист.
            Если get_best_split возвращает такое разбиение, что вся выборка относится либо к левому, либо к правому дереву,
            то когда уйдём в построение этой части, мы получаем ту же самую ситуацию. Будет зацикленная рекурсия.
            Поэтому пришлось проверять это условие здесь'''
            if self.verbose:
                print('**************\nРазбиение узла')
                print(f"best_split: {best_split}")
                
                print("*********\nУсловия на лист")  
                print(f"Условие на пустое разбиение: {condition_4}")
                print(f"{self.leafs_cnt + 1}-ый лист, значение {y[indices].mean()}")
            
            self.leafs_cnt += 1
            return Node(value = y[indices].mean())
        
        if self.bins:
            for column_idx in range(X.shape[1]):
                condition_5 = (np.any((X[indices, column_idx].min() < self.thresholds[column_idx]) & 
                                      (X[indices, column_idx].max() > self.thresholds[column_idx])))
                if condition_5:
                    # Значит, элементы X в разных бинах
                    break
            else:
                # Это лист
                print("*********\nУсловия на лист")  
                print(f"Условие на элементы в бине: {condition_5}")
                print(f"{self.leafs_cnt + 1}-ый лист, значение {y[indices].mean()}")
                
                self.leafs_cnt += 1
                return Node(value = y[indices].mean())

        # Учёто важности фичи
        self.fi[best_split[0]] += indices.size / X.shape[0] * best_split[2]

        tree_left = self.build_tree(X, y, features, indices[left_msk], depth + 1, potential + 1)
        tree_right = self.build_tree(X, y, features, indices[~left_msk], depth + 1, potential)
        
        return Node(feature=best_split[0],
                    threshold=best_split[1],
                    tree_left=tree_left,
                    tree_right=tree_right)
               

    def print_tree(self, tree: Node=None, side: str="", depth: int=1):
        if not tree:
            # Дерево не передано в качестве аргумента
            tree = self.tree
        if tree.value:
            # Это лист
            print(f"{'   '*depth}leaf_{side} = {tree.value}")
            return (1, tree.value)
        else:
            # Узел
            print(f"{depth}{'   '*depth}{tree.feature}>{tree.threshold}")
            left = self.print_tree(tree.tree_left, 'left', depth + 1)     # Отрисовка дерева слева
            right = self.print_tree(tree.tree_right, 'right', depth + 1)   # Отрисовка дерева справа

            return (left[0] + right[0], left[1] + right[1])

    def predict(self, X: np.ndarray, tree: Node=None, indices: list=None, features: list=None, prediction: np.ndarray=None) -> pd.Series:
        # При первом вызове возможна передача X как pd.DataFrame
        if isinstance(X, pd.DataFrame):
            tree = self.tree
            features = X.columns.to_list()
            indices = np.arange(X.shape[0])
            prediction = np.empty(X.shape[0])
            X = X.to_numpy()

        # Обход по дереву
        if tree.value is not None:      # Это лист
            prediction[indices] = tree.value
        else:                           # Идём по дереву
            left_msk = X[indices, features.index(tree.feature)] <= tree.threshold

            if np.any(left_msk):
                self.predict(X, tree.tree_left, indices[left_msk], features, prediction)
            if np.any(~left_msk):
                self.predict(X, tree.tree_right, indices[~left_msk], features, prediction)
        
        return pd.Series(prediction)

    def get_separators(self, X: np.ndarray):
        # Пустой список для разделителей
        self.thresholds = [None] * X.shape[1]   

        # Поиск разделителей
        for column_idx in range(X.shape[1]):
            unique_sorted = np.unique(X[:, column_idx])

            if unique_sorted.size <= self.bins:    # Ищем разделители, как среднее между уникальными значениями
                self.thresholds[column_idx] = (unique_sorted[1:] + unique_sorted[:-1]) / 2
            else:
                hist, bin_edges = np.histogram(X[:, column_idx], bins=self.bins)
                self.thresholds[column_idx] = bin_edges[1:-1]

    @staticmethod
    def mse(y: np.ndarray) -> float:
        if y.size:
            return ((y - y.mean())**2).mean()
        else:
            return 0
            