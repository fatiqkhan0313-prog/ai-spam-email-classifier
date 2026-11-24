## Summary

This project uses Artificial Intelligence and Machine Learning to classify emails as spam or not spam using TF-IDF and Logistic Regression.# AI Spam Email Classifier

A beginner-friendly AI project that classifies an email message as **spam** or **not spam** using machine learning.

## What this project does
- Converts email text into numerical features with TF-IDF.
- Trains a Logistic Regression classifier.
- Reports a test accuracy score.
- Lets you enter a message and get a spam/not-spam prediction.

> **Learning/demo project:** The included dataset is small and illustrative. It is not suitable for real-world email filtering without a much larger, carefully checked dataset and further testing.

## Requirements
- Python 3.9 or newer
- Packages listed in `requirements.txt`

## Run the project

1. Download or clone this repository.
2. Open a terminal in the project folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Train and test the model:

   ```bash
   python spam_classifier.py
   ```

5. Enter an email message when prompted. Type `exit` to stop.

## Project structure

```text
ai_spam_email_classifier/
├── data/
│   └── emails.csv
├── spam_classifier.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Example
Input:
`Congratulations! You won a free prize. Click here now`

The model returns a prediction (`spam` or `not spam`) and a spam probability.

## AI concepts used
- Supervised learning
- Text classification
- TF-IDF feature extraction
- Logistic Regression
- Training/testing split
- Model evaluation

## GitHub submission
Create a new public GitHub repository named `ai-spam-email-classifier`, upload all files in this folder, and commit the changes. Then copy the repository's main URL (format: `https://github.com/your-username/ai-spam-email-classifier`) into the course submission field.
