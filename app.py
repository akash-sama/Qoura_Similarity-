from flask import Flask, request, jsonify
import numpy as np
import pickle
from fuzzywuzzy import fuzz
import spacy

app = Flask(__name__)

with open('models/best.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

nlp = spacy.load("en_core_web_md")


def preprocess(text1, text2):
    fuzzy_ratio = fuzz.ratio(text1, text2)
    last_word_match = int(text1.split()[-1] == text2.split()[-1])

    vec1 = nlp(text1).vector
    vec2 = nlp(text2).vector
    combined_vec = np.hstack([vec1, vec2])

    features = np.hstack([fuzzy_ratio, last_word_match, combined_vec[:300]])
    return features


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text1 = data.get('text1')
    text2 = data.get('text2')

    if not text1 or not text2:
        return jsonify({'error': 'Please provide both text1 and text2'}), 400

    features = preprocess(text1, text2).reshape(1, -1)
    prediction = model.predict(features)[0]

    return jsonify({'prediction': int(prediction)})


if __name__ == '__main__':
    app.run(debug=True)
