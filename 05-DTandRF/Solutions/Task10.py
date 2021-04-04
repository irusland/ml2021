from sklearn.ensemble import RandomForestClassifier


def feature_importance(X, y):
    rf = RandomForestClassifier(n_estimators=15, max_depth=2, random_state=5,
                                criterion='entropy')
    rf.fit(X, y)
    f = np.array([])
    f_tmp = np.array([(i + 1) for i in range(len(X[0]))])
    i = rf.feature_importances_
    feature_to_value = {i[j]: f_tmp[j] for j in range(len(f_tmp))}

    sorted_dict = sorted(feature_to_value, reverse=True)
    for elem in sorted_dict:
        f = np.append(f, feature_to_value[elem])

    return f, i