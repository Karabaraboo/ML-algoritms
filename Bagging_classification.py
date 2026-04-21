import numpy as np
import pandas as pd
import random
import copy

class MyBaggingClf():
    def __init__(self,
                 estimator=None,
                 n_estimators: int=10,
                 max_samples: float=1.0,
                 random_state: int=42):
        self.estimator = estimator              # экземпляр базового класса
        self.n_estimators = n_estimators
        self.max_samples = max_samples          # доля примеров для обучайщей модели. [0.0, 1.0]
        self.random_state = random_state

    def __str__(self):
        description = []

        for name, value in self.__dict__.items():
            description.append(f"{name}={value}")

        return f"MyBaggingClf class: {', '.join(description)}"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Список для записи обученных моделей
        self.estimators = []
        
        # Количество сэмплов для обучения в модели
        rows_cnt = round(X.shape[0] * self.max_samples)

        random.seed(self.random_state)

        # Номера строк для обучения
        rows_samples = [random.choices(np.arange(X.shape[0]), k=rows_cnt) for _ in range(self.n_estimators)]

        for estimator_number in range(self.n_estimators):
            # Модель для обучения
            estimator = copy.deepcopy(self.estimator)

            # Строки матрицы фичей для обучения
            rows_sample_num = rows_samples[estimator_number]
            # Обучение
            estimator.fit(X.iloc[rows_sample_num], y.iloc[rows_sample_num])

            # Запись экземпляра
            self.estimators.append(estimator)

    def predict(self, X: pd.DataFrame, type: str) -> pd.Series:
        if type == 'mean':
            return (self.predict_proba(X) > 0.5).astype(int)
        elif type == 'vote':
            return self.predict_label(X)
        else:
            return f"Undefined type"

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        '''Метод возвращает предсказания в виде вероятностей'''
        X = X.reset_index(drop=True)
        prediction = np.zeros(X.shape[0])
        for estimator in self.estimators:
            prediction += estimator.predict_proba(X)

        return pd.Series(prediction / self.n_estimators, index=X.index)
    
    def predict_label(self, X: pd.DataFrame) -> pd.Series:
        '''Метод возвращает предсказания в виде классов'''
        X = X.reset_index(drop=True)
        prediction = np.zeros(X.shape[0])

        for estimator in self.estimators:
            prediction += estimator.predict(X) / self.n_estimators

        return pd.Series(np.where(prediction > 0.5, 1, 0), index=X.index)