from flask import Flask, request, render_template
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load the trained model and vectorizer
model = pickle.load(open("spam_classifier.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Function to clean text
def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    cleaned_message = clean_text(message)
    message_vector = vectorizer.transform([cleaned_message]).toarray()
    prediction = model.predict(message_vector)
    output = "Spam" if prediction[0] == 1 else "Not Spam"
    return render_template('index.html', prediction_text=f"Message is: {output}")

if __name__ == "__main__":
    app.run(debug=True)