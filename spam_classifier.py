import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_FILE = Path(__file__).parent / "data" / "emails.csv"


def load_data():
    texts, labels = [], []
    with DATA_FILE.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, labels


def main():
    texts, labels = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("\nEvaluation report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print("\n--- Try your own email ---")
    while True:
        message = input("Enter an email (or type 'exit'): ").strip()
        if message.lower() == "exit":
            print("Goodbye!")
            break
        if not message:
            print("Please enter some text.")
            continue

        prediction = model.predict([message])[0]
        spam_probability = model.predict_proba([message])[0][
            list(model.classes_).index("spam")
        ]
        print(f"Prediction: {prediction}")
        print(f"Estimated spam probability: {spam_probability:.1%}\n")


if __name__ == "__main__":
    main()
