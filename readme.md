
# 🧠 Keyword Extraction using TF-IDF + Word2Vec  
**Core Concept:** Hybrid NLP + Statistical Approach  
**Tech Stack:** Python, scikit-learn, gensim, NLTK, FastAPI



##  Overview
This project performs **automatic keyword extraction** from articles using a **hybrid approach** that combines:
- **TF-IDF (Term Frequency–Inverse Document Frequency):** Captures statistical importance of words across documents.  
- **Word2Vec:** Learns semantic relationships between words.

The final system allows users to send raw text to a **FastAPI endpoint** and get the most relevant keywords in response.



## 🧩 Features
✅ Preprocessing pipeline (tokenization, stopword removal, punctuation cleanup)  
✅ TF-IDF vectorization for statistical weighting  
✅ Word2Vec embedding training for semantic understanding  
✅ REST API endpoint (`/extract_keywords`) for real-time keyword extraction  
✅ Deployment-ready structure (FastAPI + Uvicorn)  


## 🏗️ Project Structure
keyword_extraction/
│
├── data/                           # Contains CSV files with categorized articles
│   ├── business_data.csv
│   ├── education_data.csv
│   ├── entertainment_data.csv
│   ├── sports_data.csv
│   └── technology_data.csv
│
├── keyword_extraction.ipynb         # Notebook for training and saving models
├── main.py                          # FastAPI app for deployment
├── tfidf_vectorizer.pkl             # Saved TF-IDF vectorizer
├── w2v_model.model                  # Saved Word2Vec model
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation


## ⚙️ Installation

### 1️⃣ Clone the Repository

git clone https://github.com/yourusername/keyword-extraction.git
cd keyword-extraction


### 2️⃣ Create and Activate Virtual Environment

py -3.12 -m venv keyword_extraction_venv
source keyword_extraction_venv/Scripts/activate   # For Windows (Git Bash)


### 3️⃣ Install Dependencies

pip install -r requirements.txt


## 🧠 Model Training (Optional)

If you want to train your own models on new articles:

1. Place your `.csv` files (containing article text) in the `data/` folder.
2. Open and run `keyword_extraction.ipynb` to preprocess data, train models, and save:

   * `tfidf_vectorizer.pkl`
   * `w2v_model.model`


## 🌐 API Deployment

### Run FastAPI Locally
uvicorn main:app --reload

Then open:
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

You’ll see the Swagger UI where you can test the endpoint `/extract_keywords`.

### Example Request

{
  "text": "Artificial intelligence is revolutionizing modern software and redefining cybersecurity."
}


### Example Response


{
  "keywords": ["artificial", "intelligence", "cybersecurity", "modern", "software"]
}

## 🧰 Technologies Used

| Category      | Tools                      |
| ------------- | -------------------------- |
| Programming   | Python 3.12                |
| NLP Libraries | NLTK, gensim, scikit-learn |
| API Framework | FastAPI                    |
| Deployment    | Uvicorn                    |
| Vector Models | TF-IDF, Word2Vec           |



## 💡 Future Improvements

* Integrate BERT embeddings for context-aware keyword extraction
* Add a front-end dashboard for visualization
* Deploy on Render or Hugging Face Spaces for public access



## 👩‍💻 Author

**Monika Chaulagain**
📫 Reach me at: chaulagainmonika2005@gmail.com



## 📜 License

This project is open-source and available under the **MIT License**.
