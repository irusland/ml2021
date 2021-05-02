from nltk.tokenize import WordPunctTokenizer, TweetTokenizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def exclude(df):
    idx = df[df['airline_sentiment'] == 'neutral'].index
    df.drop(idx, inplace = True)
    df = df.reset_index(drop=True)
    return df

    

def predict(df_train: pd.DataFrame, df_test: pd.DataFrame):
    ttk = TweetTokenizer(strip_handles=True, reduce_len=True)
    wtk = WordPunctTokenizer()
    cvr = CountVectorizer()
    
    rtk = RegexpTokenizer(r'\w+')
    
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train = exclude(df_train)
#     df_test = exclude(df_test.copy())
    
    def tok(x):
        t = ttk.tokenize(x.lower())
        t = wtk.tokenize(' '.join(t))
        t = rtk.tokenize(' '.join(t))
        return ' '.join(t)
    tr_txt = df_train['text'].apply(tok)
    print(tr_txt.head())
    ts_txt = df_test['text'].apply(tok)
    txt = tr_txt.append(ts_txt)
    
    cvr.fit_transform(txt)
    sent = df_train['airline_sentiment'].apply(lambda x: 0 if 'negative' else 1)
    
    cats = cvr.transform(tr_txt)
    print(df_train.shape)
    print(cats.shape)
    print(sent.shape)
    
    model = MultinomialNB(alpha=1, fit_prior=True)
    model.fit(cats, sent)

    return model.predict(cvr.transform(ts_txt)) 