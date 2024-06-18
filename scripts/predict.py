import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the TF-IDF vectorizer and the best Random Forest model
with open('/home/hailemichael121/Documents/hate_speech_detection/models/tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

with open('/home/hailemichael121/Documents/hate_speech_detection/models/best_rf_model_tuned.pkl', 'rb') as file:
    best_rf_model = pickle.load(file)


def predict_hate_speech(text):
    # Transform the input text
    text_transformed = tfidf_vectorizer.transform([text]).toarray()
    # Predict using the model
    prediction = best_rf_model.predict(text_transformed)
    return prediction


if __name__ == "__main__":
    sample_text = "Your sample text here."
    print(predict_hate_speech(sample_text))
