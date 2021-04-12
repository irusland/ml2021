def catfeatures(df: pd.DataFrame):
    X = df.drop('dep_delayed_15min',axis=1)
    y = df['dep_delayed_15min']
    y = y.apply(lambda x: 0 if x=='N' else 1)
    is_cat = (X.dtypes != float)
    for feature, feat_is_cat in is_cat.to_dict().items():
        if feat_is_cat:
            X[feature].fillna("NAN", inplace=True)

    cat_features_index = np.where(is_cat)[0]
            
    model = CatBoostClassifier()
    model.fit(X, y, cat_features_index)
    
    return model