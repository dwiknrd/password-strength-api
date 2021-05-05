import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename

import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
app.secret_key = b'secret'
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

#split into character
def word_split(inputs):
    character=[]
    for i in inputs:
        character.append(i)
    return character

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print(request.form['password'])
        loaded_model = pickle.load(open('model_password_strength_2.sav', 'rb'))

        loaded_vectorizer = pickle.load(open('vectorizer_password_strength_2.sav', 'rb'))
        predict_data = np.array([request.form['password']])
        prediction = loaded_vectorizer.transform(predict_data)

        result = loaded_model.predict(prediction).tolist()[0]
        
        if result == 0:
            return "Weak"
        elif result == 1:
            return "Medium"
        else:
            return "Strong"

        # return 'POST NICH!'
    return 'HELLO WORLD NIH'
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    app.run(debug=True)