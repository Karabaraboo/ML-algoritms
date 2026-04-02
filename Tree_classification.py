import numpy as np
import pandas as pd

class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature          # Параметр, по которому выполняется разбиение
        self.threshold = threshold      # Порог разбиения
        self.left = left                # Левое поддерево
        self.right = right              # Правое поддерево

        self.value = value              # вероятность первого класса, если лист


class MyTreeClf():
    def __init__(self,
                 max_depth=5,               
                 min_samples_split=2,
                 max_leafs=20,
                 verbose=False,
                 bins=None,
                 criterion='entropy'):
        self.max_depth = max_depth          # максимальная глубина дерева
        self.min_samples_split = min_samples_split   # мин допустимое кол-во объектов в листе для разбиения
        self.max_leafs = max_leafs          # макс разрешённое кол-во листов в дереве
        self.leafs_cnt = 0                  # Текущее количество листьев в дереве
        self.verbose = verbose              # Вывод на печать промежуточных результатов
        self.bins = bins                    # Количество бинов на гистограмме для фичи
        self.criterion = criterion          # Метод расчёта неопределённости (entropy или gini)
        self.fi = {}                        # Словарь для учёта важностей фичей (feature importance)

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

        # Исходная неопределённость
        S0 = getattr(self, self.criterion)(y)

        #print(f"S0 = {S0}")
        for column in X:
            if self.bins:
                separators = self.thresholds[column]
            else:    
                sorted_values = np.sort(X[column].unique())
                separators = (sorted_values[:-1] + sorted_values[1:]) / 2

            for sep in separators:
                #print(f"Слева {(X[column] <= sep).sum()} элементов, из них {X[column].loc[(X[column] <= sep) & (y == 0)].count()} нулевого класса")
                left_idx = X[column] <= sep     # bool-индексы элементов слева
                right_idx = X[column] > sep     # bool-индексы элементов справа
                left_num = left_idx.sum()       # количество элементов слева
                right_num = right_idx.sum()     # и справа
                S_left = getattr(self, self.criterion)(y[left_idx])
                S_right = getattr(self, self.criterion)(y[right_idx])
                
                #print(f"S1 = {S_left}, S2 = {S_right}")
                # Прирост информации
                ig_current = S0 - left_num / (left_num + right_num) * S_left - right_num / (left_num + right_num) * S_right

                #print(f"current column is {column}, current sep = {sep}, current ig = {ig_current}")
                if ig_current > self.ig:
                    self.ig = ig_current
                    self.col_name = column
                    self.split_value = sep
                
                #print(f"best column is {self.col_name}, best ig = {self.ig}")
        
        return (self.col_name, self.split_value, self.ig)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Сохраняю названия фичей, т.к. в build_tree используется numpy
        column_names = X.columns.tolist()
        X = X.to_numpy()
        y = y.to_numpy()

        # Если bins задано, то определяем границы бинов гистограммы
        if self.bins:
            self.thresholds = [None]*X.shape[1]     # Пустой список длиной, равной количеству фичей
            self.get_separators(X, y)

        # Обнуление на случай повторного вызова fit
        self.leafs_cnt = 0 
        self.fi = {}
        for col_name in column_names:
            self.fi[col_name] = 0

        self.tree = self.build_tree(X=X,
                                    y=y,
                                    indices=np.arange(X.shape[0]),
                                    depth=1, 
                                    features=column_names)

    def build_tree(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray, depth: int, features: str):
        if self.verbose:
            print(f"indices = {indices}")
            print('********* \n условия в build_tree')
            print(f"\ndepth = {depth}")
            print(f'X.shape[0] < 2: {X[indices].shape[0] < 2}')
            print(f"y.sum() == y.shape[0]: {y[indices].sum() == y[indices].shape[0]}, y.sum() == 0: {y[indices].sum() == 0}")
            print(f"depth > self.max_depth: {depth > self.max_depth}")
            print(f"y.shape[0] < self.min_samples_split: {y[indices].shape[0] < self.min_samples_split}")
            print(f"self.leafs_cnt > self.max_leafs: {self.leafs_cnt > self.max_leafs}")
            print(f"self.leafs_cnt + 1 >= self.max_leafs: {self.leafs_cnt + 1 >= self.max_leafs}")

        # Если это лист
        if (X[indices].shape[0] < 2 or                                              # Если выборка содержит 1 элемент
            y[indices].sum() == y[indices].shape[0] or y[indices].sum() == 0 or     # или в ней один класс
            depth > self.max_depth or                                               # превышена допустимая глубина дерева
            y[indices].shape[0] < self.min_samples_split or                         # число элементов меньше минимально допустимого
            self.leafs_cnt + 1 >= self.max_leafs and depth > 1):                    # число листов больше допустимого
            # Последнее условие - число созданных листьев и потенциальных (по одному на каждый уровень выше)

            # тогда это лист
            self.leafs_cnt += 1

            if self.verbose:
                print("\n********Построение нового узла")
                print(f"\n Число листьев = {self.leafs_cnt}, значение листа: {np.mean(y[indices])}")

            # тогда возращаем вероятность первого класса
            return Node(value = np.mean(y[indices]))
        
        # Проверка на лист по критерию попадания в один бин
        if self.bins:
            for col_number in range(X.shape[1]):
                # Если хотя бы для одной фиче найдутся элементы, которые будут между границами бинов, то продолжаем
                if np.any((self.thresholds[col_number] >= X[indices, col_number].min()) &
                (self.thresholds[col_number] <= X[indices, col_number].max())):
                    break
            else:
                # Если таких элементов нет (всё в одном бине), то это лист
                return Node(value = np.mean(y[indices]))

        # Если это узел, то разбиваем      
        best_split = self.get_best_split(pd.DataFrame(X[indices]), pd.Series(y[indices]))
        if self.verbose:
            print("\n*********Разделение узла")
            print(f"Массив X:\n{X}\nbest_split: {best_split}")

        self.fi[features[best_split[0]]] += indices.shape[0] / X.shape[0] * best_split[2]

        left_msk = X[indices, best_split[0]] <= best_split[1]
        
            
        '''создание ветвления добавляет один потенциальный лист при заходе в левую ветку,
        но при заходе в правую - потенциальное количество листов прежнее
        Т.е. для левой ветки количество потенциальных листьев = depth, 
        но для правой = depth - 1'''
        left_tree = self.build_tree(X, y, indices[left_msk], depth + 1, features)
        right_tree = self.build_tree(X, y, indices[~left_msk], depth + 1, features)
        
        # Запись в Node
        return Node(feature=features[best_split[0]],
                    threshold=best_split[1],
                    left=left_tree,
                    right=right_tree)
    
    def print_tree(self):
        return self.printing_node(self.tree)

    def printing_node(self, node: Node, side: str=None, depth: int=0):
        if node.value is not None:
            # if self.verbose:
            print(f"{'   ' * depth} leaf_{side} = {node.value}")

            return (1, node.value)
        else:
            # if self.verbose:
            print(f"{'   ' * depth} {node.feature} > {node.threshold}:")

            left = self.printing_node(node.left, side='left', depth=depth + 1)
            right = self.printing_node(node.right, side='right', depth=depth + 1)
            n_leafs = left[0] + right[0]
            sum_leafs = left[1] + right[1]
            return (n_leafs, sum_leafs)
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        return 1 * (self.predict_proba(X) > 0.5)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        # Пустой массив для записи предсказаний
        prediction = np.empty(X.shape[0], dtype=float)
        indices = np.arange(X.shape[0])
        self.traverse_tree(X, self.tree, prediction, indices)

        return pd.Series(prediction)
        
    def traverse_tree(self, X: pd.DataFrame, tree: Node, prediction: np.ndarray, indices: np.ndarray) -> np.ndarray:
        # prediction - это один и тот же объект в памяти на каждой итерации, т.к. передача по ссылке
        if tree.value is not None:
            prediction[indices] = tree.value
        else:
            # маска по условию
            left_msk = X.iloc[indices][tree.feature] <= tree.threshold

            if np.any(left_msk):
                # Просто передавать тот же X, а не выделять в нём строки - лучше, из-за передачи по ссылке
                self.traverse_tree(X, tree.left, prediction, indices[left_msk])
            if np.any(~left_msk):
                self.traverse_tree(X, tree.right, prediction, indices[~left_msk])

    def get_separators(self, X: np.ndarray, y: np.ndarray):
        # если количество строк в X больше, чем bins, используем bins
        # X.shape[0] = количеству разделителей + 1
        if X.shape[0] > self.bins:
            # используем гистограмму
            for column_number in range(X.shape[1]):
                hist, bin_edges = np.histogram(X[:, column_number], bins=self.bins)
                self.thresholds[column_number] = bin_edges[1:-1]
        else:
            for column_number in range(X.shape[1]):
                sorted_values = np.unique(X[:, column_number])
                self.thresholds[column_number] = (sorted_values[:-1] + sorted_values[1:]) / 2

    @staticmethod
    def entropy(y):
        if y.size:
            p1 = y.sum() / y.shape[0]
        else:
            p1 = 0
        p0 = 1 - p1
        return 0 if p0 * p1 == 0 else -p0 * np.log2(p0) - p1 * np.log2(p1)
    
    @staticmethod
    def gini(y):
        if y.size:
            p1 = y.sum() / y.shape[0]
        else:
            p1 = 0
        p0 = 1 - p1
        return 1 - (p0**2 + p1**2)
    
