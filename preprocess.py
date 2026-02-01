# preprocess.py
import pandas as pd
import re
import nltk
import joblib
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logging.info("🚀 Starting preprocessing...")

# ---------------- NLTK SETUP ----------------
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# ---------------- LOAD DATA ----------------
try:
    df = pd.read_csv("movies.csv")
    logging.info("✅ Dataset loaded successfully. Rows: %d", len(df))
except Exception as e:
    logging.error("❌ Failed to load dataset: %s", str(e))
    raise e

# ---------------- TEXT PREPROCESSING ----------------
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# ---------------- SELECT REQUIRED COLUMNS ----------------
required_columns = ["genres", "keywords", "overview", "title"]
df = df[required_columns].dropna().reset_index(drop=True)

# ---------------- COMBINE TEXT ----------------
df["combined"] = (
    df["genres"] + " " +
    df["keywords"] + " " +
    df["overview"]
)

logging.info("🧹 Cleaning text...")
df["cleaned_text"] = df["combined"].apply(preprocess_text)
logging.info("✅ Text cleaned.")

# ---------------- TF-IDF VECTORIZATION ----------------
logging.info("🔠 Vectorizing text using TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
tfidf.fit(df["cleaned_text"])
logging.info("✅ TF-IDF vectorizer trained.")

# ---------------- SAVE ONLY SMALL FILES ----------------
joblib.dump(df, "df_cleaned.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

logging.info("💾 Saved df_cleaned.pkl & tfidf_vectorizer.pkl")
logging.info("✅ Preprocessing complete.")
