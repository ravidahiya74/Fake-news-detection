import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

news = pd.read_csv('news.csv')
news.drop('Unnamed: 0', axis=1, inplace=True)
news['title_length'] = news['title'].apply(len)
news['text_length'] = news['text'].apply(len)
X = news['text']
y = news['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                    ('passiveaggressive', PassiveAggressiveClassifier(max_iter=50))])
                    
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

from sklearn.externals import joblib
filename = 'finalized_model.sav'
joblib.dump(pipeline, filename)
with open('model.pickle', 'wb') as handle:
    pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)