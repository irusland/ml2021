from sklearn.base import BaseEstimator, ClassifierMixin


def calculate_probability(y: np.ndarray):
    probs = np.array([])
    _, unique_counts = np.unique(y, return_counts=True)
    for count in unique_counts:
        probs = np.append(probs, count / len(y))
    return probs


def _entropy(y: np.ndarray) -> float:
    probs = calculate_probability(y)
    s = -np.sum(probs * np.log2(probs))
    return s


def _gini_impurity(y: np.ndarray) -> float:
    probs = calculate_probability(y)
    return 1 - np.sum(np.power(probs, 2))


class MyDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=4, criterion='entropy'):
        self.max_depth = max_depth
        self.criterion = criterion
        self.tree = {}
        self._criteria_func = {
            'gini': _gini_impurity,
            'entropy': _entropy
        }

    def inform_gain(self, X: np.ndarray, y: np.ndarray,
                    threshold: float) -> float:
        criteria_func = self._criteria_func[self.criterion]
        return (criteria_func(y) - criteria_func(y[X <= threshold]) * len(
            y[X <= threshold]) / len(y)
                - criteria_func(y[X > threshold]) * len(y[X > threshold]) / len(
                    y))

    def get_best_threshold(self, X: np.ndarray, y: np.ndarray) -> (
    float, float):
        assert X.ndim == 1
        assert y.ndim == 1
        best_threshold = 0
        best_score = 0
        for threshold in X:
            score = self.inform_gain(X, y, threshold)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        return best_threshold, best_score

    def find_best_split(self, X, y):
        assert X.ndim == 2
        assert y.ndim == 1
        best_feature = 0
        best_score = 0
        best_threshold = 0

        for feature in range(len(X[0])):
            threshold, score = self.get_best_threshold(X[:, feature], y)
            if score > best_score:
                best_feature, best_score, best_threshold = feature, score, threshold

        return best_feature, best_threshold, best_score

    def _build_tree(self, X, y, depth):
        split_feature, split_value, split_score = self.find_best_split(X, y)

        if split_score < 0.01 or depth >= self.max_depth:
            return {'value': (split_feature, split_value),
                    'leaf': True,
                    'left': None,
                    'right': None,
                    'prob': len(y[y == 1]) / len(y)
                    }

        left_inds = X[:, split_feature] <= split_value
        right_inds = X[:, split_feature] > split_value

        left_tree = self._build_tree(X[left_inds], y[left_inds], depth + 1)
        right_tree = self._build_tree(X[right_inds], y[right_inds], depth + 1)

        return {'value': (split_feature, split_value),
                'leaf': False,
                'left': left_tree,
                'right': right_tree,
                'prob': len(y[y == 1]) / len(y)
                }

    def _predict_sample(self, node, x, depth):
        feature, value = node["value"]

        if node["leaf"] or depth >= self.max_depth:
            return node['prob']

        if x[feature] <= value:
            return self._predict_sample(node["left"], x, depth + 1)
        return self._predict_sample(node["right"], x, depth + 1)


    def fit(self, X: np.ndarray, y: np.ndarray):
        self.tree = self._build_tree(X, y, depth=0)
        return self


    def predict_proba(self, X: np.ndarray):
        res = []
        for x in X:
            i = self._predict_sample(self.tree, x, 0)
            res.append([1 - i, i])
        return np.array(res)


    def predict(self, X: np.ndarray):
        res = []
        for x in X:
            res.append(round(self._predict_sample(self.tree, x, 0)))
        return np.array(res)