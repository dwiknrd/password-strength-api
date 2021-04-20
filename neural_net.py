import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import AdaBoostClassifier

#read the document
data = pd.read_csv('data.csv', error_bad_lines=False)

#missing value
data[data['password'].isnull()]
data.dropna(inplace=True)

#Convert our entire data into format of numpy array
password_tuple = np.array(data)

#random shuffle
import random
random.shuffle(password_tuple)

x = [labels[0] for labels in password_tuple]
y = [labels[1] for labels in password_tuple]


#split into character
def word_split(inputs):
  character=[]
  for i in inputs:
    character.append(i)
  return character

#Apply TD-IDF on data
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(tokenizer=word_split)
X = vectorizer.fit_transform(x)
vectorizer.get_feature_names()

first_vector_document = X[0]
tf_idf = pd.DataFrame(first_vector_document.T.todense(), index=vectorizer.get_feature_names(), columns=['TF-IDF'])
print(tf_idf.sort_values(by=['TF-IDF'], ascending=False))

#Built logistic regression model

#split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scores_test = []
scores_train = []
n_estimators = []
for n_est in range(30):
    ada = AdaBoostClassifier(n_estimators = n_est + 1, random_state = 42)
    ada.fit(X_train, y_train)
    n_estimators.append(n_est + 1)
    scores_test.append(ada.score(X_test, y_test))
    scores_train.append(ada.score(X_train, y_train))
# Our Ada Boost score on our train set.
ada.score(X_train, y_train)
# Our Ada Boost score on our test set.
ada.score(X_test, y_test)

# Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix, accuracy_score
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# accuracy_score(y_test, y_pred)