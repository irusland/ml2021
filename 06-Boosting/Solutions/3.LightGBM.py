from lightgbm import LGBMRegressor

def lgbmreg(X_train, y_train):
    return LGBMRegressor(max_depth=5, learning_rate = .02).fit(X_train, y_train)