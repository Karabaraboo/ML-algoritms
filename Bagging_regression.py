import numpy as np
import pandas as pd
import random
import copy

class MyBaggingReg():
    def __init__(self,
                 estimator=None,
                 n_estimators: int=10,
                 max_samples: float=1.0,
                 random_state: int=42):
        self.estimator = estimator          # экземпляр базового класса
        self.n_estimators = n_estimators    # количество базовых экземпляров
        self.max_samples = max_samples      # доля сэмплов для обучения одного экземпляра
        self.random_state = random_state

    def __str__(self):
        description = []

        for key, value in self.__dict__.items():
            description.append(f"{key}={value}")

        return f"MyBaggingReg class: " + ", ".join(description)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Все модели
        self.estimators = []

        rows_sample_cnt = round(X.shape[0] * self.max_samples)
        random.seed(self.random_state)

        # Список из n_estimators списков с номерами строк X
        rows_samples = [random.choices(np.arange(X.shape[0]), k=rows_sample_cnt) for estimator_number in range(self.n_estimators)]

        for estimator_number in range(self.n_estimators):
            # Индексы строк X для обучения
            sample_rows_idx = rows_samples[estimator_number]
            
            # Копия модели
            instance_estimator = copy.deepcopy(self.estimator) 
            #print(type(instance_estimator))
            instance_estimator.fit(X.iloc[sample_rows_idx], y.iloc[sample_rows_idx])
            self.estimators.append(instance_estimator)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        prediction = np.zeros(X.shape[0])

        for estimator in self.estimators:
            prediction += estimator.predict(X) / self.n_estimators

        return prediction
    
