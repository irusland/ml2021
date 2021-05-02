from nltk.tokenize import WordPunctTokenizer, TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def predict(df_train: pd.DataFrame, df_test: pd.DataFrame):
    ttk = TweetTokenizer()
    vectorizer = CountVectorizer(tokenizer=lambda x: x,
                                 preprocessor=lambda x: x)

    X_train = df_train['text'].apply(ttk.tokenize)
    X_train = vectorizer.fit_transform(X_train)

    X_test = df_test['text'].apply(ttk.tokenize)
    X_test = vectorizer.transform(X_test)

    y_train = df_train['airline_sentiment'].apply(
        lambda x: 1 if 'positive' else 0)

    model = MultinomialNB()
    model.fit(X_train, y_train)


    return model.predict(X_test)