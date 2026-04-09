# ==========================================
# FAKE NEWS DETECTION SYSTEM
# Using Logistic Regression and Naive Bayes
# ==========================================

# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import re
import string
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample


print("\nLoading data...\n")

# ==========================================
# LOAD DATA FROM SINGLE CSV FILE
# ==========================================
try:
    df = pd.read_csv("news.csv")  # Replace with your dataset name
    print("Data loaded successfully!\n")
except FileNotFoundError:
    print("Error: 'news.csv' file not found!")
    exit()

# Ensure required columns exist
if "label" not in df.columns:
    raise ValueError("Dataset must contain a 'label' column.")

# Handle missing columns
if "title" not in df.columns:
    df["title"] = ""

if "text" not in df.columns:
    raise ValueError("Dataset must contain either 'text' or both 'title' and 'text' columns.")

# Keep only necessary columns
df = df[["title", "text", "label"]]

# ==========================================
# BALANCE DATASET
# ==========================================
print("Balancing dataset...\n")

fake = df[df.label == 0]
real = df[df.label == 1]

min_samples = min(len(fake), len(real))

fake_balanced = resample(fake,
                         replace=False,
                         n_samples=min_samples,
                         random_state=42)

real_balanced = resample(real,
                         replace=False,
                         n_samples=min_samples,
                         random_state=42)

df = pd.concat([fake_balanced, real_balanced])

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Label distribution after balancing:\n")
print(df["label"].value_counts())

# ==========================================
# DATA CLEANING
# ==========================================
print("\nCleaning text...\n")

df.dropna(inplace=True)
df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.replace("\n", " ")
    return text.strip()

df["content"] = df["content"].apply(clean_text)

# ==========================================
# SPLIT DATA
# ==========================================
X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# TF-IDF VECTORIZATION
# ==========================================
print("Vectorizing text...\n")

tfidf = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    min_df=2,
    ngram_range=(1, 2),
    max_features=15000
)

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# ==========================================
# TRAIN MODELS
# ==========================================
print("Training models...\n")

# Logistic Regression
lr = LogisticRegression(max_iter=2000, class_weight='balanced')
lr.fit(X_train_vec, y_train)
lr_pred = lr.predict(X_test_vec)

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
nb_pred = nb.predict(X_test_vec)

# ==========================================
# RESULTS
# ==========================================
print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))
print("Classification Report:\n", classification_report(y_test, lr_pred))

print("\n--- Naive Bayes ---")
print("Accuracy:", accuracy_score(y_test, nb_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, nb_pred))
print("Classification Report:\n", classification_report(y_test, nb_pred))

# ==========================================
# SELECT BEST MODEL
# ==========================================
lr_score = accuracy_score(y_test, lr_pred)
nb_score = accuracy_score(y_test, nb_pred)

if lr_score >= nb_score:
    model = lr
    model_name = "Logistic Regression"
else:
    model = nb
    model_name = "Naive Bayes"

print(f"\nBest Model Selected: {model_name}")

# ==========================================
# PREDICTION FUNCTION
# ==========================================
def predict_news(text):
    text = clean_text(text)
    vec = tfidf.transform([text])
    pred = model.predict(vec)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(vec).max()
        label = "REAL" if pred == 1 else "FAKE"
        return f"{label} ({prob:.2f})"
    else:
        return "REAL" if pred == 1 else "FAKE"

# ==========================================
# SAMPLE TEST
# ==========================================
print("\nSample Test:")
print("Result:", predict_news("India's economy shows steady growth"))

# ==========================================
# USER INPUT
# ==========================================
print("\nType your own news (type 'exit' to stop)\n")

while True:
    user_text = input("Enter news: ")
    if user_text.lower() == "exit":
        break
    print("Result:", predict_news(user_text))
    print("-" * 50)

# ==========================================
# SAVE MODEL AND VECTORIZER
# ==========================================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))

print("\nModel and vectorizer saved successfully!")

# ==========================================
# VERIFY SAVED MODEL
# ==========================================
loaded_model = pickle.load(open("model.pkl", "rb"))
loaded_vec = pickle.load(open("vectorizer.pkl", "rb"))

test_text = "The economy is growing rapidly"
vec = loaded_vec.transform([clean_text(test_text)])
pred = loaded_model.predict(vec)[0]

print("\nLoaded Model Test:",
      "REAL" if pred == 1 else "FAKE")

print("\nDone.")
