import numpy as np

def bias_variance_decomp(x_test:np.array, y_test:int, estimators:list)->tuple:
	M = len(estimators)
	mean = sum([model.predict(x_test) for model in estimators]) / M
	error = sum([(model.predict(x_test) - y_test)**2 for model in estimators]) / M
	bias2 = (y_test - mean)**2
	variance = sum([(model.predict(x_test) - mean)**2 for model in estimators]) / M
	error, bias2, variance = error[0], bias2[0], variance[0]
	return error, bias2, variance