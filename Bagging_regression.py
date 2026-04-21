import numpy as np
import pandas as pd
import random
import copy

class MyBaggingReg():
    def __init__(self,
                 estimator=None,
                 n_estimators: int=10,
                 max_samples: float=1.0,
                 oob_score=None,
                 random_state: int=42):
        self.estimator = estimator          # экземпляр базового класса
        self.n_estimators = n_estimators    # количество базовых экземпляров
        self.max_samples = max_samples      # доля сэмплов для обучения одного экземпляра
        self.oob_score = oob_score          # метрика для оценки oob-score (mae, mse, rmse, mape, r2)
        self.oob_score_ = 0                 # значение oob-error
        self.random_state = random_state

    def __str__(self):
        description = []

        for key, value in self.__dict__.items():
            description.append(f"{key}={value}")

        return f"MyBaggingReg class: " + ", ".join(description)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Все модели
        self.estimators = []

        # Обнуление на случай повтороного запуска fit
        self.oob_score_ = 0

        rows_sample_cnt = round(X.shape[0] * self.max_samples)
        random.seed(self.random_state)

        # Список из n_estimators списков с номерами строк X
        rows_samples = [random.choices(np.arange(X.shape[0]), k=rows_sample_cnt) for estimator_number in range(self.n_estimators)]

        if self.oob_score:
            y_predict = np.zeros(X.shape[0])        # Запись предсказаний на oob-выборке
            estimator_cnt = np.zeros(X.shape[0])    # Количество моделей для oob-выборки

        for estimator_number in range(self.n_estimators):
            # Индексы строк X для обучения
            sample_rows_idx = rows_samples[estimator_number]
            
            # Копия модели
            instance_estimator = copy.deepcopy(self.estimator) 
            #print(type(instance_estimator))
            instance_estimator.fit(X.iloc[sample_rows_idx], y.iloc[sample_rows_idx])
            self.estimators.append(instance_estimator)

            if self.oob_score:
                oob_msk = ~np.isin(np.arange(X.shape[0]), sample_rows_idx)      # строки, которые в oob-выборке

                y_predict[oob_msk] += instance_estimator.predict(X.iloc[oob_msk]).to_numpy()
                estimator_cnt[oob_msk] += 1

        if self.oob_score:
            y_predict = np.divide(y_predict, estimator_cnt, out=np.full_like(y_predict, np.nan), where=estimator_cnt != 0)
            
            # Расчёт метрики
            oob_score_msk = ~np.isnan(y_predict)
            self.oob_score_ = getattr(self, self.oob_score)(y_predict[oob_score_msk], y[oob_score_msk])

    def predict(self, X: pd.DataFrame) -> pd.Series:
        prediction = np.zeros(X.shape[0])

        for estimator in self.estimators:
            prediction += estimator.predict(X) / self.n_estimators

        return prediction
    
    # Метрики для подсчёта OOB-error
    @staticmethod
    def mse(y_prediction, y_real, n=None) -> float:
        if isinstance(y_prediction, pd.Series):
            y_prediction = y_prediction.to_numpy()
        if isinstance(y_real, pd.Series):
            y_real = y_real.to_numpy()
        if not n:
            n = len(y_prediction)
        return 1 / n * np.sum((y_prediction - y_real)**2)
    
    @staticmethod
    def mae(y_prediction, y_real, n=None) -> float:
        if isinstance(y_prediction, pd.Series):
            y_prediction = y_prediction.to_numpy()
        if isinstance(y_real, pd.Series):
            y_real = y_real.to_numpy()
        if not n:
            n = len(y_prediction)
        return 1 / n * np.sum(np.abs(y_real - y_prediction))

    @staticmethod
    def rmse(y_prediction, y_real, n=None) -> float:
        if isinstance(y_prediction, pd.Series):
            y_prediction = y_prediction.to_numpy()
        if isinstance(y_real, pd.Series):
            y_real = y_real.to_numpy()
        if not n:
            n = len(y_prediction)
        return np.sqrt(1 / n * np.sum((y_real - y_prediction)**2))
    
    @staticmethod
    def mape(y_prediction, y_real, n=None) -> float:
        if isinstance(y_prediction, pd.Series):
            y_prediction = y_prediction.to_numpy()
        if isinstance(y_real, pd.Series):
            y_real = y_real.to_numpy()
        if not n:
            n = len(y_prediction)
        return 100 / n * np.sum(np.abs((y_real - y_prediction) / y_real))
    
    @staticmethod
    def r2(y_prediction, y_real, n=None) -> float:
        if isinstance(y_prediction, pd.Series):
            y_prediction = y_prediction.to_numpy()
        if isinstance(y_real, pd.Series):
            y_real = y_real.to_numpy()
        if not n:
            n = len(y_prediction)
        return 1 - np.sum((y_real - y_prediction)**2) / np.sum((y_real - y_real.mean())**2)
