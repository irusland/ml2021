def entropy(y: np.ndarray) -> float:
    probs = calculate_probability(y)
    s = -np.sum(probs * np.log2(probs))
    return s


def gini_impurity(y: np.ndarray) -> float:
    probs = calculate_probability(y)
    return 1 - np.sum(np.power(probs, 2))


def inform_gain(X: np.ndarray, y: np.ndarray, threshold: float,
                criteria_func=entropy) -> float:
    X = [(X[i], y[i]) for i in range(len(X))]
    X1 = np.array([])
    X2 = np.array([])
    y1 = np.array([])
    y2 = np.array([])
    tmp1 = []

    tmp2 = []

    for i in range(len(X)):
        if X[i][0] >= threshold:
            tmp1.append((X[i][0], X[i][1]))
        else:
            tmp2.append((X[i][0], X[i][1]))

    for i in range(len(tmp1)):
        X1 = np.append(X1, tmp1[i][0])
        y1 = np.append(y1, tmp1[i][1])

    for i in range(len(tmp2)):
        X2 = np.append(X2, tmp2[i][0])
        y2 = np.append(y2, tmp2[i][1])

    return criteria_func(y) - (len(X1) / len(X)) * criteria_func(y1) - (
                len(X2) / len(X)) * criteria_func(y2)


def get_best_threshold(X: np.ndarray, y: np.ndarray, criteria_func=entropy) -> (
float, float):
    assert X.ndim == 1
    assert y.ndim == 1
    best_threshold = np.median(X)
    best_score = criteria_func(y)
    return best_threshold, best_score


def find_best_split(X, y, criteria_func=entropy):
    X = X.T
    assert X.ndim == 2
    assert y.ndim == 1
    best_feature = 0
    best_score = 0
    best_threshold = 0

    for i in range(len(X)):
        best_threshold, _ = get_best_threshold(X[i], y)
        current_gain = inform_gain(X[i], y, best_threshold, entropy)
        print(f'current_gain: {current_gain}')
        if current_gain > best_score:
            best_score = current_gain
            best_feature = i

        best_threshold, _ = get_best_threshold(X[best_feature], y)
    return best_feature, best_threshold, best_score


def calculate_probability(y: np.ndarray):
    probs = np.array([])
    _, unique_counts = np.unique(y, return_counts=True)
    for count in unique_counts:
        probs = np.append(probs, count / len(y))
    return probs