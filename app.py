from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Load NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

def clean_text(text):
    if isinstance(text, str) is False:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Load saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    prediction = model.predict(X)[0]

    return jsonify({
        "input": text,
        "cleaned": cleaned,
        "sentiment": prediction
    })

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Flask sentiment API running"})
@app.route("/ui", methods=["GET", "POST"])
def ui():
    sentiment = None

    if request.method == "POST":
        user_text = request.form.get("user_text", "")
        cleaned = clean_text(user_text)
        X = vectorizer.transform([cleaned])
        sentiment = model.predict(X)[0]

    return """
        <html>
            <head>
                <title>Sentiment Analyzer</title>
            </head>
            <body>
                <h2>Sentiment Analyzer</h2>
                <form method="POST">
                    <textarea name="user_text" rows="4" cols="50" placeholder="Enter text here"></textarea><br><br>
                    <button type="submit">Analyze</button>
                </form>
                <h3>{}</h3>
            </body>
        </html>
    """.format(sentiment if sentiment is not None else "")


if __name__ == "__main__":
    app.run()
