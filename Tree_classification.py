import numpy as np
import pandas as pd

class MyTreeClf():
    def __init__(self,
                 max_depth=5,               # максимальная глубина дерева
                 min_samples_split=2,       # мин допустимое кол-во объектов в листе для разбиения
                 max_leafs=20):             # макс разрешённое кол-во листов в дереве
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
    
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

        for column in X:
            sorted_values = np.sort(X[column].unique())
            separators = (sorted_values[:-1] + sorted_values[1:]) / 2

            # Исходная энтропия
            p0, p1 = X[column].groupby(y).count() / X.shape[0]
            S0 = -p0 * np.log2(p0) - p1 * np.log2(p1) 
            for sep in separators:
                # Порядок False - 0, False - 1, True - 0, True - 1
                p11, p12, p21, p22 = X[column].groupby([X[column] > sep, y]).count() / X.shape[0]
                
                if p11 * p12 == 0:
                    S1 = 0
                else:
                    S1 = -p11 * np.log2(p11) - p12 * np.log2(p12)
                if p21 * p22 == 0:
                    S2 = 0
                else:
                    S2 = -p21 * np.log2(p21) - p22 * np.log2(p22)
                
                # Прирост информации
                ig_current = S0 - S1 - S2

                if ig_current >= self.ig:
                    self.ig = ig_current
                    self.col_name = column
                    self.split_value = sep
        
        


