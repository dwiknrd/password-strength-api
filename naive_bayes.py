import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train)

#prediction test set
y_pred = clf.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))

#accuracy= 0,77