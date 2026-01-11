class MyLogReg():
    # Логистическая регрессия
    def __init__(self,
                 n_iter=10,
                 learning_rate=0.1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None     # Хранение весов модели
    
    def __str__(self):
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
