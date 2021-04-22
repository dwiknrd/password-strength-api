import pickle
import pandas as pd
import numpy as np

loaded_model = pickle.load(open('model_password_strength.sav', 'rb'))


#split into character
def word_split(inputs):
  character=[]
  for i in inputs:
    character.append(i)
  return character

from sklearn.feature_extraction.text import TfidfVectorizer


loaded_vectorizer = pickle.load(open('vectorizer_password_strength.sav', 'rb'))

predict_data = np.array(['12345678'])
prediction = loaded_vectorizer.transform(predict_data)
print(loaded_model.predict(prediction))