import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from gensim import utils
import gensim.parsing.preprocessing as gsp
from sklearn.model_selection import GridSearchCV



def clean_text(s): #remove some trash from text
    filters = [
        gsp.strip_tags,
        gsp.strip_punctuation,
        gsp.strip_multiple_whitespaces,
        gsp.strip_numeric,
        gsp.remove_stopwords,
        gsp.stem_text
    ]
    s = s.lower()
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
    return s




df_test = pd.read_csv('df_test')
df_train = pd.read_csv('df_train')


df_test = shuffle(df_test, random_state=13)
df_train = shuffle(df_train, random_state=13)

#prepare train
X_train = df_train.loc[:, 'text']
X_train = X_train.apply(lambda x: clean_text(x))
X_train = X_train.to_numpy()
X_ltext_train = df_train.loc[:, 'l_text']
y_train = df_train.loc[:, 'rating'].to_numpy() - 1 #1-10 to 0-9
y_train[y_train >= 6] -= 2 #6-9 to 4-7
y_train_bin = df_train.loc[:, 'estimate'].to_numpy()

#prepare test
X_test = df_test.loc[:, 'text']
X_test = X_test.apply(lambda x: clean_text(x))
X_test = X_test.to_numpy()
y_test = df_test.loc[:, 'rating'].to_numpy() - 1 #1-10 to 0-9
y_test[y_test >= 6] -= 2 #6-9 to 4-7
y_test_bin = df_test.loc[:, 'estimate'].to_numpy()


print(np.min(y_train), np.max(y_train), np.min(y_test), np.max(y_test))

#word to vec
vectorizer_tv = TfidfVectorizer(ngram_range=(1,3), min_df=50)
vectorizer_tv = vectorizer_tv.fit(X_train)
X_train = vectorizer_tv.transform(X_train)
X_test = vectorizer_tv.transform(X_test)
print('shape X_train ', X_train.shape)


parameters = {
    'max_depth': range(2, 8, 1),
    'reg_alpha': [0, 0.5, 1],
    'reg_lambda': [0, 0.5, 1]}



xgb_model = XGBClassifier(seed=17, booster='gbtree', objective='multi:softprob')


grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=parameters,
    scoring='accuracy',
    cv=2,
    verbose=3)


grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
df_gs = pd.DataFrame(grid_search.cv_results_)
df_gs.to_csv('grid_search.csv')








