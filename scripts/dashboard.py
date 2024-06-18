import streamlit as st
import pickle
import nltk
from PIL import Image
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the pre-trained model and vectorizer


def load_model():
    with open('/home/hailemichael121/Documents/hate_speech_detection/models/logistic_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


def load_vectorizer():
    with open('/home/hailemichael121/Documents/hate_speech_detection/models/tfidf_vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    return vectorizer


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [
        word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


# Load model and vectorizer
model = load_model()
vectorizer = load_vectorizer()

# Define classes
classes = ['Non-Hate', 'Hate', 'Neutral']

# Streamlit configuration
st.set_page_config(page_title="Hate Speech Detection Dashboard", page_icon="📢")

# Dashboard title
st.title("Hate Speech Detection Dashboard")

# Landing page content
st.markdown("""
## Welcome to the Hate Speech Detection Dashboard

This dashboard allows you to analyze and visualize the results of hate speech detection on various text inputs. You can explore the dataset, see the evaluation metrics, and even input your own text to check for hate speech.

### Overview
Hate speech detection is a crucial task in maintaining healthy and safe online environments. Our model uses advanced natural language processing techniques and machine learning algorithms to detect hate speech with high accuracy.

### Objectives
- **Analyze** the distribution of hate speech in our dataset.
- **Visualize** the results of our machine learning models.
- **Interact** with the model by inputting text to check for hate speech.
""")

image_path = "/home/hailemichael121/Documents/hate_speech_detection/plots/HateSpeechImage.jpeg"
st.image(image_path, caption="Hate Speech Image", use_column_width=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio(
    "Go to", ["Home", "EDA Analysis", "Model Evaluation", "Hate Speech Detection"])

# Home Page
if options == "Home":
    st.markdown("""
    ### About This Project
    This project aims to build a robust hate speech detection system using machine learning. The system is trained on a large dataset of text inputs labeled as hate speech or non-hate speech.

    ### Dataset
    The dataset used in this project contains text data from various social media platforms. Each text entry is labeled as hate speech or non-hate speech. The data has been preprocessed to remove noise and irrelevant information.

    ### Preprocessing
    The text data undergoes several preprocessing steps:
    - Tokenization
    - Removing stopwords
    - Lemmatization

    ### Model
    The model used for hate speech detection is a Logistic Regression model. It has been trained on the preprocessed text data using TF-IDF vectorization.

    ### Future Work
    Future enhancements include improving the model accuracy, expanding the dataset, and integrating the system with real-time data streams.
    """)

# EDA Analysis Page
elif options == "EDA Analysis":
    st.markdown("## EDA Analysis Results")
    st.markdown("""
    Exploratory Data Analysis (EDA) helps us understand the dataset and identify patterns and insights.

    ### Distribution of Hate Speech
    This chart shows the distribution of hate speech versus non-hate speech in the dataset.

    ### Text Length Distribution
    Analyzing the length of text entries can provide insights into the nature of hate speech.

    ### Top 20 Words
    The most frequently occurring words in hate speech and non-hate speech texts.

    ### Word Cloud
    A visual representation of the most common words in the dataset.
    """)

    images_eda = [
        '/home/hailemichael121/Documents/hate_speech_detection/plots/HatespeechVsNonhatespeechDist.png',
        '/home/hailemichael121/Documents/hate_speech_detection/plots/textLengthDist.png',
        '/home/hailemichael121/Documents/hate_speech_detection/plots/top20.png',
        '/home/hailemichael121/Documents/hate_speech_detection/plots/WordCloud.png'
    ]
    descriptions_eda = [
        'Hate Speech Vs Non Hate Speech distribution',
        'Text Length Distribution',
        'Top 20 words',
        'Word cloud'
    ]

    for i in range(len(images_eda)):
        with st.container():
            image = Image.open(images_eda[i])
            st.image(image, caption=descriptions_eda[i], use_column_width=True)

# Model Evaluation Page
elif options == "Model Evaluation":
    st.markdown("## Model Evaluation Results")
    st.markdown("""
    Evaluating the performance of our model is crucial to ensure its reliability.

    ### Confusion Matrix
    The confusion matrix shows the performance of the classification model.

    ### ROC Curve
    The ROC curve illustrates the model's ability to distinguish between classes.

    ### Evaluation Metrics
    The classification report provides detailed metrics such as precision, recall, and F1-score.
    """)

    images_evaluation = [
        '/home/hailemichael121/Documents/hate_speech_detection/plots/confusionMatrix.png',
        '/home/hailemichael121/Documents/hate_speech_detection/plots/ROC.png'
    ]
    descriptions_evaluation = [
        'Confusion Matrix',
        'ROC Curve'
    ]

    for i in range(len(images_evaluation)):
        with st.container():
            image = Image.open(images_evaluation[i])
            st.image(
                image, caption=descriptions_evaluation[i], use_column_width=True)

    st.header("Analysis and Visualizations")

    selected_option = st.selectbox(
        "Select an option", ["Confusion Matrix", "ROC Curve", "Evaluation Metrics"])
    st.write(f"You selected: {selected_option}")

    def plot_confusion_matrix(y_true, y_pred, classes):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        st.pyplot(fig)

    def plot_roc_curve(y_true, y_probs, n_classes):
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc[i] = roc_auc_score(y_true_bin[:, i], y_probs[:, i])

        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_true_bin.ravel(), y_probs.ravel())
        roc_auc["micro"] = roc_auc_score(y_true_bin, y_probs, average="micro")

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = roc_auc_score(y_true_bin, y_probs, average="macro")

        fig, ax = plt.subplots()
        plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(
            roc_auc["micro"]), color='deeppink', linestyle=':', linewidth=4)
        plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'.format(
            roc_auc["macro"]), color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        st.pyplot(fig)

    # Replace with actual test labels
    y_test = np.random.randint(0, 3, size=1000)
    # Replace with actual predicted labels
    y_test_pred = np.random.randint(0, 3, size=1000)
    # Replace with actual predicted probabilities for each class
    y_test_probs = np.random.rand(1000, 3)
    classes = ['Hate', 'Non-Hate', 'Neutral']

    if selected_option == "Confusion Matrix":
        plot_confusion_matrix(y_test, y_test_pred, classes)
    elif selected_option == "ROC Curve":
        plot_roc_curve(y_test, y_test_probs, len(classes))
    elif selected_option == "Evaluation Metrics":
        report = classification_report(
            y_test, y_test_pred, target_names=classes, output_dict=True)
        df = pd.DataFrame(report).transpose()
        st.dataframe(df)

# Hate Speech Detection Page
elif options == "Hate Speech Detection":
    st.markdown("## Check Text for Hate Speech")
    st.markdown("""
    Enter text in the box below to check if it contains hate speech. The model will analyze the text and provide a prediction.

    ### Example
    Try inputting different types of text to see how the model responds. For instance:
    - "I hate you."
    - "You are amazing!"
    - "Go back to where you came from."

    ### Instructions
    1. Type or paste your text in the input box.
    2. Click the 'Check' button.
    3. See the prediction results below.
    """)

    user_input = st.text_area("Enter text here:")
    if st.button("Check"):
        preprocessed_text = preprocess_text(user_input)
        vectorized_text = vectorizer.transform([preprocessed_text])
        prediction = model.predict(vectorized_text)
        prediction_proba = model.predict_proba(vectorized_text)

        st.markdown(f"**Prediction:** {classes[prediction[0]]}")
        st.markdown("**Prediction Probabilities:**")
        for i, label in enumerate(classes):
            st.markdown(f"- **{label}**: {prediction_proba[0][i]:.2f}")

st.sidebar.header("About")
st.sidebar.markdown(
    """
    This dashboard is developed for the purpose of detecting hate speech on social media platforms. 
    It leverages advanced machine learning techniques to provide reliable and accurate predictions.
    
    ### Contact
    - **Developers**: Group 2
    - **Section**: D
   
    
    For more information, feel free to reach out!
    """
)
