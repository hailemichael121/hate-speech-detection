# Hate Speech Detection System

This project is a Hate Speech Detection System built using Python and various machine learning libraries.

## Project Structure

- `data/`: Directory containing raw and processed data.
- `models/`: Directory where trained models and vectorizers are saved.
- `notebooks/`: Jupyter notebooks for exploration and development.
- `scripts/`: Python scripts for data preparation, model training, evaluation, and deployment.
- `venv/`: Virtual environment for project dependencies.

## Setup

1. **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd hate_speech_detection
    ```

2. **Create and activate virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run data preparation:**

    ```bash
    python scripts/data_preparation.py
    ```

5. **Train the model:**

    ```bash
    python scripts/model_training.py
    ```

6. **Evaluate the model:**

    ```bash
    python scripts/model_evaluation.py
    ```

7. **Run the dashboard:**

    ```bash
    streamlit run scripts/dashboard.py
    ```

8. **Run the API:**

    ```bash
    python scripts/api.py
    ```

## Usage

- **Training:** To train the model with the provided dataset.
- **Evaluation:** To evaluate the performance of the trained model.
- **Prediction API:** To use the trained model for predicting hate speech in new text.

## License

This project is licensed under the MIT License.
