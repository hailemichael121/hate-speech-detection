import pandas as pd
import numpy as np
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Function to load and concatenate datasets


def load_data(file_paths):
    dataframes = [pd.read_csv(file) for file in file_paths]
    return pd.concat(dataframes, ignore_index=True)

# Function to clean text


def clean_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans(
        '', '', string.punctuation))  # Remove punctuation
    text = text.strip()  # Remove leading/trailing whitespace
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word not in stopwords.words(
        'english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word)
              for word in tokens]  # Lemmatize tokens
    return ' '.join(tokens)


# Main script
if __name__ == "__main__":
    raw_data_paths = [
        '/home/hailemichael121/Documents/hate_speech_detection/data/raw/HateSpeechDataSet1.csv',
        '/home/hailemichael121/Documents/hate_speech_detection/data/raw/HateSpeechDataSet2.csv',
        '/home/hailemichael121/Documents/hate_speech_detection/data/raw/HateSpeechDataSet3.csv'
    ]

    # Load and concatenate data
    df = load_data(raw_data_paths)

    # Clean text data
    df['cleaned_text'] = df['tweet'].apply(clean_text)

    # Handle missing values (simple approach: drop rows with missing values)
    df.dropna(subset=['tweet', 'cleaned_text', 'class'], inplace=True)

    # Save processed data
    processed_data_dir = '/home/hailemichael121/Documents/hate_speech_detection/data/processed/'
    os.makedirs(processed_data_dir, exist_ok=True)
    train_path = os.path.join(processed_data_dir, 'train.csv')
    test_path = os.path.join(processed_data_dir, 'test.csv')

    # Assuming a simple split for train/test data
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
