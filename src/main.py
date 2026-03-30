# ==============================
# IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample


print("\nLoading data...\n")

# ==============================
# LOAD DATA
# ==============================
try:
    fake_data = pd.read_csv("Fake.csv")
    real_data = pd.read_csv("True.csv")

    fake_data["label"] = 0
    real_data["label"] = 1

    df = pd.concat([fake_data, real_data])
    print("Data loaded successfully!\n")

except:
    print("Error: Dataset files not found!")
    exit()

# ==============================
# BALANCE DATASET (VERY IMPORTANT)
# ==============================
print("\nBalancing dataset...\n")

fake = df[df.label == 0]
real = df[df.label == 1]

# downsample fake to match real
fake = resample(fake,
                replace=False,
                n_samples=len(real),
                random_state=42)

df = pd.concat([fake, real])

# shuffle
df = df.sample(frac=1).reset_index(drop=True)

print("Label distribution after balancing:\n", df["label"].value_counts())

# ==============================
# CLEANING DATA
# ==============================
df.dropna(inplace=True)

df["content"] = df["title"] + " " + df["text"]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.replace("\n", " ")
    return text

print("\nCleaning text...\n")
df["content"] = df["content"].apply(clean_text)

# ==============================
# SPLIT DATA
# ==============================
X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# TF-IDF (IMPROVED)
# ==============================
print("\nVectorizing text...\n")

tfidf = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    min_df=2,
    ngram_range=(1, 2),
    max_features=15000
)

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# ==============================
# TRAIN MODELS
# ==============================
print("\nTraining models...\n")

# Logistic Regression (balanced)
lr = LogisticRegression(max_iter=2000, class_weight='balanced')
lr.fit(X_train_vec, y_train)
lr_pred = lr.predict(X_test_vec)

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
nb_pred = nb.predict(X_test_vec)

# ==============================
# RESULTS
# ==============================
print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

print("\n--- Naive Bayes ---")
print("Accuracy:", accuracy_score(y_test, nb_pred))
print(confusion_matrix(y_test, nb_pred))
print(classification_report(y_test, nb_pred))

# ==============================
# SELECT BEST MODEL
# ==============================
lr_score = accuracy_score(y_test, lr_pred)
nb_score = accuracy_score(y_test, nb_pred)

model = lr if lr_score > nb_score else nb
print("\nUsing:", "Logistic Regression" if model == lr else "Naive Bayes")

# ==============================
# PREDICTION FUNCTION
# ==============================
def predict_news(text):
    text = clean_text(text)
    vec = tfidf.transform([text])

    pred = model.predict(vec)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(vec).max()
        return ("REAL ✅" if pred == 1 else "FAKE ❌") + f" ({prob:.2f})"
    else:
        return "REAL ✅" if pred == 1 else "FAKE ❌"

# ==============================
# SAMPLE TEST
# ==============================
print("\nSample test:")
print("Result:", predict_news("India's economy shows steady growth"))

# ==============================
# USER INPUT
# ==============================
print("\nType your own news (type 'exit' to stop)\n")

while True:
    print("TO TEST THIS MODDEL PLEASE USE TEST CASES PROVIDED IN THE project DESCRIPTION , PROJECT REPORT AND ALSO IN README.md. THIS IS DUE TO  FINITE DATASET AND LIMITED TRAINING DATA  IS AVLIABLE ON DIFFERENT SITES ")
    user_text = input("Enter news: ")

    if user_text.lower() == "exit":
        break

    print("Result:", predict_news(user_text))
    print("-" * 40)

# ==============================
# SAVE MODEL
# ==============================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))

print("\nModel saved successfully!")

# ==============================
# VERIFY
# ==============================
loaded_model = pickle.load(open("model.pkl", "rb"))
loaded_vec = pickle.load(open("vectorizer.pkl", "rb"))

test_text = "Economy is growing rapidly"
vec = loaded_vec.transform([test_text])
pred = loaded_model.predict(vec)

print("\nLoaded model test:",
      "REAL ✅" if pred[0] == 1 else "FAKE ❌")

print("\nDone 👍")
