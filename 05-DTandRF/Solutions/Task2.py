def gini_impurity(y: np.ndarray) -> float:
    probs = calculate_probability(y)
    return 1 - np.sum(np.power(probs, 2))


def entropy(y: np.ndarray) -> float:
    probs = calculate_probability(y)
    s = -np.sum(probs * np.log2(probs))
    return s


def inform_gain(X: np.ndarray, y: np.ndarray, threshold: float,
                criteria_func=entropy) -> float:
    value_to_bin = {X[i]: y[i] for i in range(len(X))}
    condition = X <= threshold
    X1 = X[condition]
    X2 = X[~condition]

    convert = lambda x: value_to_bin[x]
    y1 = np.array(list(map(convert, X1)))
    y2 = np.array(list(map(convert, X2)))

    return criteria_func(y) - (len(X1) / len(X)) * criteria_func(y1) - (
                len(X2) / len(X)) * criteria_func(y2)


def get_best_threshold(X: np.ndarray, y: np.ndarray, criteria_func=entropy) -> (
float, float):
    assert X.ndim == 1
    assert y.ndim == 1
    best_threshold = np.median(X)
    best_score = criteria_func(y)
    return best_threshold, best_score


def calculate_probability(y: np.ndarray):
    probs = np.array([])
    _, unique_counts = np.unique(y, return_counts=True)
    for count in unique_counts:
        probs = np.append(probs, count / len(y))
    return probs