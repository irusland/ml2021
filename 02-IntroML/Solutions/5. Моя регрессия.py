import numpy as np

# Нашел крутой видос про то, как найти вектор коэффициентов для регрессии
# Что самое важное -- в общем виде

# https://www.youtube.com/watch?v=szU45TcpYHI

# А вот тут вывод этого уравнения
# https://habr.com/ru/company/ods/blog/322076/

# В связи с этим пришлось немного пошаманить с размерностями и транспонированием

class LinReg():
    def __init__(self):
        self.coef_ = None
        self.theta = None

    def fit(self, X_train: np.array, y_train: np.array):
        X_train = X_train.T
        X_train = np.vstack((np.ones_like(X_train), X_train))
        y_train = np.array([y_train]).T
        one = np.linalg.pinv(np.dot(X_train, X_train.T))
        two = np.dot(X_train, y_train)
        theta = np.dot(one.T, two)

        self.theta = theta
        self.coef_ = np.concatenate(theta.T)[::-1]
        return self

    def predict(self, X_test: np.array):
        X_test = X_test.T
        X_test = np.vstack((np.ones_like(X_test), X_test))
        y_pred = np.dot(X_test.T, self.theta).T
        return np.concatenate(y_pred)