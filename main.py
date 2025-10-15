import pickle
from gensim.models import Word2Vec
from fastapi import FastAPI
from pydantic import BaseModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import numpy as np

# NLTK setup
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load saved models
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
w2v_model = Word2Vec.load("w2v_model.model")
feature_names = vectorizer.get_feature_names_out()

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [w for w in word_tokenize(text) if w not in stop_words]
    return tokens

# Keyword extraction
def get_top_keywords(tokens, top_n=5):
    word_scores = {}
    tfidf_vector = vectorizer.transform([" ".join(tokens)])
    for word in tokens:
        if word in feature_names:
            tfidf_index = list(feature_names).index(word)
            tfidf_score = tfidf_vector[0, tfidf_index]
            word_scores[word] = tfidf_score
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_words[:top_n]]

def extract_keywords_new_article(text, top_n=5):
    tokens = preprocess_text(text)
    return get_top_keywords(tokens, top_n)

# FastAPI app
app = FastAPI()

class Article(BaseModel):
    text: str

@app.post("/extract_keywords")
def get_keywords(article: Article):
    keywords = extract_keywords_new_article(article.text)
    return {"keywords": keywords}
