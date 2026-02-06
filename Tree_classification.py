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
    