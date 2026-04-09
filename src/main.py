
# IMPORT LIBRARIES

import pandas as pd
import re
import string
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample

print("\nLoading datasets...\n")


# LOAD DATASETS

try:
    fake_news_df = pd.read_csv("Fake.csv")
    true_news_df = pd.read_csv("True.csv")

    # Add labels
    fake_news_df["label"] = 0  # Fake News
    true_news_df["label"] = 1  # Real News

    # Combine datasets
    news_df = pd.concat([fake_news_df, true_news_df], ignore_index=True)
    print("Datasets loaded successfully!\n")

except FileNotFoundError:
    print("Error: 'Fake.csv' and 'True.csv' not found!")
    exit()


# BALANCE THE DATASET

print("Balancing dataset...\n")

fake_news = news_df[news_df["label"] == 0]
real_news = news_df[news_df["label"] == 1]

min_count = min(len(fake_news), len(real_news))

fake_news_balanced = resample(
    fake_news,
    replace=False,
    n_samples=min_count,
    random_state=42
)

real_news_balanced = resample(
    real_news,
    replace=False,
    n_samples=min_count,
    random_state=42
)

balanced_df = pd.concat([fake_news_balanced, real_news_balanced])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Label Distribution:")
print(balanced_df["label"].value_counts(), "\n")


# TEXT CLEANING FUNCTION

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)          # Remove URLs
    text = re.sub(r"<.*?>", "", text)            # Remove HTML tags
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)              # Remove numbers
    text = text.replace("\n", " ")               # Remove newlines
    return text.strip()

print("Cleaning text data...\n")

# Remove missing values
balanced_df.dropna(subset=["title", "text"], inplace=True)

# Combine title and text
balanced_df["content"] = (
    balanced_df["title"] + " " + balanced_df["text"]
).apply(preprocess_text)


# SPLIT DATA

features = balanced_df["content"]
labels = balanced_df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)


# TF-IDF VECTORIZATION

print("Vectorizing text...\n")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    min_df=2,
    ngram_range=(1, 2),
    max_features=15000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# TRAIN MODELS

print("Training models...\n")

logistic_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)
logistic_model.fit(X_train_vec, y_train)
lr_predictions = logistic_model.predict(X_test_vec)

naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train_vec, y_train)
nb_predictions = naive_bayes_model.predict(X_test_vec)


# EVALUATE MODELS

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, lr_predictions))
print(confusion_matrix(y_test, lr_predictions))
print(classification_report(y_test, lr_predictions))

print("\n--- Naive Bayes ---")
print("Accuracy:", accuracy_score(y_test, nb_predictions))
print(confusion_matrix(y_test, nb_predictions))
print(classification_report(y_test, nb_predictions))


# SELECT BEST MODEL

lr_accuracy = accuracy_score(y_test, lr_predictions)
nb_accuracy = accuracy_score(y_test, nb_predictions)

best_model = logistic_model if lr_accuracy >= nb_accuracy else naive_bayes_model
best_model_name = (
    "Logistic Regression" if best_model == logistic_model else "Naive Bayes"
)

print(f"\nBest Model Selected: {best_model_name}")


# PREDICTION FUNCTION

def predict_news(news_text):
    cleaned_text = preprocess_text(news_text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = best_model.predict(vectorized_text)[0]

    if hasattr(best_model, "predict_proba"):
        probability = best_model.predict_proba(vectorized_text).max()
        return ("REAL" if prediction == 1 else "FAKE") + f" ({probability:.2f})"
    else:
        return "REAL" if prediction == 1 else "FAKE"


# SAMPLE TEST

print("\nSample Prediction:")
print("Result:", predict_news("India's economy shows steady growth"))


# USER INPUT

print("\nEnter your own news (type 'exit' to quit):\n")

while True:
    user_input = input("Enter news: ")
    if user_input.lower() == "exit":
        break
    print("Result:", predict_news(user_input))
    print("-" * 50)


# SAVE MODEL AND VECTORIZER

pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel saved successfully!")


# VERIFY SAVED MODEL

loaded_model = pickle.load(open("model.pkl", "rb"))
loaded_vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

test_text = "Economy is growing rapidly"
test_vec = loaded_vectorizer.transform([test_text])
prediction = loaded_model.predict(test_vec)

print("\nLoaded Model Test:",
      "REAL" if prediction[0] == 1 else "FAKE")

print("\nDone!")
