import pandas as pd
import numpy as np
import re
import pickle
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import os

# Download stopwords if not downloaded
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load dataset (Ensure spam.csv is in the same folder as the script)
file_path = os.path.join(os.path.dirname(__file__), "spam.csv")
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep necessary columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels to numerical values (ham=0, spam=1)
df["label"] = LabelEncoder().fit_transform(df["label"])

# Function to clean text
def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

df["message"] = df["message"].apply(clean_text)

# Convert text to features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["message"]).toarray()
y = df["label"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model and vectorizer
with open("spam_classifier.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Model training complete! The model and vectorizer are saved.")
