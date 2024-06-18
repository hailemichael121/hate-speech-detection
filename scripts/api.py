from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the TF-IDF vectorizer and the best Random Forest model
with open('/home/hailemichael121/Documents/hate_speech_detection/models/tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

with open('/home/hailemichael121/Documents/hate_speech_detection/models/best_rf_model_tuned.pkl', 'rb') as file:
    best_rf_model = pickle.load(file)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Transform the input text
    text_transformed = tfidf_vectorizer.transform([text]).toarray()
    # Predict using the model
    prediction = best_rf_model.predict(text_transformed)
    return jsonify({'prediction': prediction[0]})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
