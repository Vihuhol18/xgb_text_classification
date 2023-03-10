import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from gensim import utils
import gensim.parsing.preprocessing as gsp
from sklearn.metrics import recall_score, accuracy_score, average_precision_score, f1_score



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
print(np.unique(y_train), np.unique(y_test))

#words to vec
vectorizer_tv = TfidfVectorizer(ngram_range=(1,3), min_df=50) #selection of ngram with frequency greater than 50
vectorizer_tv = vectorizer_tv.fit(X_train)
X_train = vectorizer_tv.transform(X_train)
X_test = vectorizer_tv.transform(X_test)
print('shape X_train ', X_train.shape)



xgb_model = XGBClassifier(seed=17, booster='gbtree', objective='multi:softprob',
                          max_depth=6, reg_alpha=1, reg_lambda=1, verbosity=2)


xgb_model.fit(X_train, y_train)
predict = xgb_model.predict_proba(X_test)
print(predict.shape)
predict = np.argmax(predict, axis=-1)
print('\n rating recall', recall_score(y_test, predict, average='weighted', zero_division=0),
      # '\n rating precision', average_precision_score(y_test, predict, average='weighted'),
      '\n rating f1', f1_score(y_test, predict, average='weighted', zero_division=0))
predict[predict < 4] = 0
predict[predict >= 4] = 1
print('\n estimate recall', recall_score(y_train_bin, predict),
      '\n estimate acc', accuracy_score(y_train_bin, predict),
      '\n estimate precision', average_precision_score(y_train_bin, predict),
      '\n estimate f1', f1_score(y_train_bin, predict))
xgb_model.save_model('xgb_model.json')

