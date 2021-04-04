def gini_impurity(y: np.ndarray) -> float:
    probs = calculate_probability(y)
	return 1 - np.sum(np.power(probs, 2))

def entropy(y: np.ndarray) -> float:
    probs = calculate_probability(y)
    s = np.sum(probs * np.log2(probs))
    return -s

def calc_criteria(y: np.ndarray) -> (float, float):
    assert y.ndim == 1
    return entropy(y), gini_impurity(y)

def calculate_probability(y: np.ndarray):
    probs = np.array([])
    _, unique_counts = np.unique(y, return_counts=True)
    for count in unique_counts:
        probs = np.append(probs,count / len(y))
    return probs