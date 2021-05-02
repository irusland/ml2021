def boost_step(estimator, weights, X_train, y_train, eps = 1e-6):
	estimator.fit(X_train, y_train)
	y_pred = estimator.predict(X_train)
	weights /= np.sum(weights)
	error = np.sum(np.where(y_train != y_pred, weights, 0))
	alpha = .5 * np.log((1 - error + eps) / (error + eps))
	new_weights = weights * np.exp(-alpha * y_pred * y_train) / np.sum(weights * np.exp(-alpha * y_pred * y_train))
    return y_pred, error, alpha, new_weights