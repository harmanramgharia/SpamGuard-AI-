# SpamGuard AI — Complete KNN Spam Detection Web App

This is a complete Streamlit project. It automatically downloads a public copy of the Kaggle SMS Spam Collection CSV on first launch, so you do not need to manually download or upload the dataset.

## Dataset used

Kaggle: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

The dataset has labelled messages:
- `ham` = legitimate message
- `spam` = unwanted message

A public CSV mirror is used by the app to fetch the Kaggle dataset automatically. Internet access is required the first time the app loads.

## Run it

1. Extract `SpamGuard_KNN_Complete.zip`.
2. Open the extracted folder in VS Code.
3. Open the terminal in that folder.
4. Install the libraries:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the web app:

   ```bash
   streamlit run app.py
   ```

6. The app opens in your browser. Enter a message and click **Detect spam**.

## Features

- Automatic dataset loading
- Text preprocessing with TF-IDF
- KNN classifier (`K=5`, cosine distance)
- Spam / Not Spam prediction
- Training and testing split (80:20)
- Accuracy, confusion matrix and classification report
- Dataset sample and class distribution

## Project files

- `app.py` — complete web app and machine-learning code
- `requirements.txt` — Python dependencies
- `data/` — optional location for a local `spam.csv`

## Important academic note

The Kaggle SMS Spam Collection is a dataset of SMS messages, not full email records with email headers and attachments. Therefore, the accurate project description is **SMS Spam Detection Using KNN**. It classifies message text entered by the user.
