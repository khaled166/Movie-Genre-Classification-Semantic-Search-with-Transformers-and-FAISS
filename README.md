# 🎬 Movie Genre Prediction & Semantic Retrieval Engine

End-to-end NLP + ML pipeline for multi-label movie genre prediction, plot summarization using HuggingFace Transformers, and semantic retrieval powered by FAISS and sentence embeddings. Combines classical ML, modern NLP, and vector search for intelligent media analysis and recommendation.

---

## 📌 Project Highlights

- 🔍 **Genre Prediction** using multi-label classification (Logistic Regression, Random Forest, XGBoost, LightGBM, etc.).
- 🧠 **Text Summarization** of movie plots using Hugging Face Transformers (`facebook/bart-large-cnn`).
- 📚 **Vector Embeddings** of plot summaries using `sentence-transformers`.
- 🔎 **Semantic Search Engine** using **FAISS** for fast and scalable similarity retrieval.
- 📈 Advanced **EDA**, word clouds, and genre correlation insights.
- 💡 Designed for research, real-world deployment, and retrieval-augmented generation (RAG) scenarios.

---

## 📂 Dataset Overview

This project uses a dataset of **34,886 movies** with the following key features:

- 🎞️ `Title`
- 📆 `Release Year`
- 🌍 `Origin/Ethnicity`
- 🎬 `Director`
- 🌟 `Cast`
- 🧠 `Plot Summary`
- 🧾 `Genres` (Multi-label)
- 🔗 `Wikipedia Link`

---

## ⚙️ Tech Stack

| Category             | Tools & Libraries                                                                 |
|----------------------|-----------------------------------------------------------------------------------|
| Data Manipulation    | `pandas`, `numpy`, `ast`, `json`                                                  |
| EDA & Visualization  | `seaborn`, `matplotlib`, `wordcloud`, `plotly.express`                            |
| NLP & Preprocessing  | `TfidfVectorizer`, `re`, `sentence-transformers`, `transformers`, `evaluate`      |
| ML Models            | `sklearn`, `XGBoost`, `LightGBM`, `MLPClassifier`, `OneVsRestClassifier`          |
| Vector Search        | `FAISS` by Facebook AI                                                             |
| Text Summarization   | Hugging Face Pipeline (`facebook/bart-large-cnn`)                                 |
| Evaluation Metrics   | `accuracy_score`, `hamming_loss`, `ROUGE`, `classification_report`                |
| Optimization         | `GridSearchCV`, `RandomizedSearchCV`, `SMOTE`, `pandarallel`, `swifter`           |

---

## 🧪 Main Features & Notebooks

| Module | Description |
|--------|-------------|
| `EDA & Preprocessing` | Clean and explore metadata, cast, genres, and plot summaries. |
| `Genre Prediction` | Multi-label classification using TF-IDF and various classifiers. |
| `Summarization` | Use transformer models to generate concise versions of long movie plots. |
| `Vector DB & Retrieval` | Encode plots with `sentence-transformers`, index with FAISS, and retrieve similar movies. |
| `Evaluation` | Use multi-label classification metrics and ROUGE for summarization performance. |

---

## 🚀 How to Run

1. Clone this repo:
    ```bash
    git clone https://github.com/yourusername/movie-genre-prediction
    cd movie-genre-prediction
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the notebook or scripts:
    ```bash
    jupyter notebook Movies_Genre_Prediction.ipynb
    ```

4. Optional: Try semantic search via FAISS cell in notebook.

---

## 🔮 Potential Extensions

- Deploy the retrieval engine via Streamlit or Gradio.
- Add recommendation system with collaborative filtering.
- Fine-tune BERT-based models for better summarization.
- Integrate RAG for movie Q&A (e.g., "Find movies similar to *Inception* with a strong female lead").

---

## 🧠 Author

**Khaled SeifAldin**  
Machine Learning & NLP Engineer  
[GitHub](https://github.com/khaled166) • [Kaggle](https://www.kaggle.com/khaledseif166) • [LinkedIn](https://www.linkedin.com/in/khaled-seifaldin-089a7a1b7/) • [Linktree](https://linktr.ee/khaledseif166)

---

## 📄 License

This project is licensed under the MIT License - feel free to use and modify for research and educational purposes.

---

## 💬 Feedback & Collaboration

If you find this useful or want to collaborate on similar projects (NLP, ML, semantic search, recommendation systems), feel free to connect on LinkedIn or reach out via Linktree!

