from collections import defaultdict
from math import log
import numpy as np


class MyNaiveBayes():
    def __init__(self, alpha=1):
        self.class_freq = defaultdict(lambda: 0)
        self.feat_freq = defaultdict(lambda: 0)
        self.aplha = alpha

    def fit(self, X, y):
        assert X.ndim == 2
        assert y.ndim == 1
        for feature, label in zip(X, y):
            self.class_freq[label] += 1
            for value in feature:
                self.feat_freq[(value, label)] += 1

        for k in self.class_freq:
            self.class_freq[k] /= len(X)

        for value, label in self.feat_freq:
            self.feat_freq[(value, label)] /= self.class_freq[label]

        return self

    def predict(self, X: np.ndarray):
        assert X.ndim == 2
        y_pred = []
        for i in range(len(X)):
            y_pred.append(min(self.class_freq.keys(),
                   key=lambda cls: self.calc_freq(X[i], cls)))
        return y_pred

    def calc_freq(self, X, cls):
        freq = - log(self.class_freq[cls])
        for feature in X:
            freq += - log(self.feat_freq.get((feature, cls), 10 ** (-7)))
        return freq
