# preprocess.py
import pandas as pd
import re
import nltk
import joblib
import logging
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# ---------------- NLTK ----------------
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# ---------------- LOAD DATA ----------------
df = pd.read_csv("movies.csv")

required_columns = ["genres", "keywords", "overview", "title"]
df = df[required_columns].dropna().reset_index(drop=True)

# ---------------- CLEAN TEXT ----------------
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    return " ".join([w for w in tokens if w not in stop_words])

df["combined"] = df["genres"] + " " + df["keywords"] + " " + df["overview"]
df["cleaned_text"] = df["combined"].apply(preprocess_text)

# ---------------- TF-IDF ----------------
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df["cleaned_text"])

# ---------------- TOP-K COSINE SIM ----------------
TOP_K = 20
logging.info("📐 Computing TOP-%d cosine similarity...", TOP_K)

cosine_sim_topk = {}

cosine_sim = cosine_similarity(tfidf_matrix).astype(np.float32)

for idx in range(cosine_sim.shape[0]):
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:TOP_K+1]
    cosine_sim_topk[idx] = sim_scores

# ---------------- SAVE FILES ----------------
joblib.dump(df, "df_cleaned.pkl")
joblib.dump(cosine_sim_topk, "cosine_sim.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

logging.info("💾 Saved reduced cosine_sim.pkl")
logging.info("✅ Preprocessing complete.")

