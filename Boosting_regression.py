import numpy as np
import pandas as pd

from Tree_regression import *
'''
В бустинге применяют чаще всего решающие деревья. Теоретически можно и другие ансамбли, но деревья работают лучше

Обычно для градиентного бустинга выбирают не глубокие деревья (~6-8).
В то время как для случайного леса наоборот - строят глубокие деревья (~16).
'''

class MyBoostReg():
    def __init__(self,
                 n_estimators: int=10,
                 learning_rate: float = 0.1,
                 max_depth: int=5,
                 min_samples_split: int=2,
                 max_leafs: int=20,
                 bins: int=16,
                 verbose=False):
        # Параметры бустинга
        self.n_estimators = n_estimators        # количество деревьев в лесу
        self.learning_rate = learning_rate      # скорость обучения

        # Параметры деревьев
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins

        # Другие параметры
        self.pred_0 = pd.Series()               # начальное предсказание
        self.trees = [None]*self.n_estimators   # список обученных деревьев

        self.verbose = verbose

    def __str__(self):
        description = []

        for name, value in self.__dict__.items():
            description.append(f"{name}={value}")

        return f"MyBoostReg class: {', '.join(description)}"
    
    def fit(self, X: pd.DataFrame, y: pd.Series):    
        self.pred_0 = y.mean()

        if self.verbose:
            print(self.pred_0)

        # для нулевого дерева
        predictions = self.pred_0
        # Цикл по количеству деревьев
        for tree_number in range(self.n_estimators):
            # отклонения от истинных значений для предыдущего дерева
            current_error = y - predictions

            tree = MyTreeReg(max_depth=self.max_depth,
                             min_samples_split=self.min_samples_split,
                             max_leafs=self.max_leafs,
                             bins=self.bins)
            tree.fit(X, current_error)

            # Сохраняем обученное дерево
            self.trees[tree_number] = tree

            # Новое предсказание
            predictions += self.learning_rate * tree.predict(X)
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        prediction = pd.Series(np.zeros(X.shape[0]))

        for tree in self.trees:
            prediction += tree.predict(X)

        return self.pred_0 + self.learning_rate * prediction

        