# scripts/model_training.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from imblearn.pipeline import Pipeline as imbPipeline
from imblearn.over_sampling import RandomOverSampler
from joblib import dump
import os

# Load preprocessed data
train_data = pd.read_csv('data/processed/train.csv')
test_data = pd.read_csv('data/processed/test.csv')

# Define feature and target
X_train = train_data['cleaned_text']
y_train = train_data['class']
X_test = test_data['cleaned_text']
y_test = test_data['class']

# Implement checkpointing
checkpoint_dir = 'model_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, 'model_checkpoint.joblib')

# Define the model pipeline
pipeline = imbPipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('oversample', RandomOverSampler()),
    ('clf', LogisticRegression(max_iter=1000))
])

try:
    # Train the model with checkpointing
    print("Training the model...")
    pipeline.fit(X_train, y_train)

    # Save the model checkpoint
    dump(pipeline, checkpoint_path)
    print(f"Model checkpoint saved at {checkpoint_path}")

    # Validation
    y_val_pred = pipeline.predict(X_test)
    print("Validation Accuracy:", accuracy_score(y_test, y_val_pred))
    print("Validation Classification Report:\n",
          classification_report(y_test, y_val_pred))

except Exception as e:
    print("An error occurred during training:", str(e))
