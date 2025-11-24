# 🎬 TinyLlama Movie Chatbot (Local RAG App)

<p align="center">
  <a href="https://www.youtube.com/watch?v=zful6iXCnm4">
    <img src="https://img.youtube.com/vi/zful6iXCnm4/0.jpg" width="600" />
  </a>
</p>


This project is a fully local Retrieval-Augmented Generation (RAG) chatbot for movie data, built on:

- 🦙 TinyLlama (local 1.1B parameter LLM)
- 💾 Chroma vector store
- 🔎 LangChain for retrieval and prompt orchestration
- 🧠 Sentence-transformer embeddings

Ask questions like:

✅ *"Give me movie review for Inception"*  
✅ *"List movie titles with IMDB ratings 8"*  
✅ *"List movie titles with genres 'Drama','Thriller'"*  
✅ *"List movies directed by Christopher Nolan"*  

All answers are grounded in your local dataset of movies!

---

## 💻 Features

- Runs fully **offline** on your GPU (or CPU).
- Uses your **local** TinyLlama model (or any HF causal LM).
- Embeds your CSV data into a Chroma vector store.
- Lets you chat naturally about movie ratings, genres, directors, overviews.

---

## 📦 Project Structure

├── vector_store_builder.py # Script to create your Chroma DB from CSV
├── app.py # Chatbot app to query your local LLM
├── Movies_Reviews_modified_version1.csv # Your movie dataset
└── README.md # You're reading it!


---

## ⚡️ Dataset Format

The Dataset is taken from Kaggle: IMDB Movies Dataset for top 1000 movies by IMDB Rating.
Your CSV should have these columns:

- `Series_Title` — Movie name
- `Released_Year` — Year of release
- `Genre` — Comma-separated genres
- `IMDB_Rating` — Numeric rating
- `Overview` — Short plot description
- `Director` — Director's name

Example row:

| Series_Title | Released_Year | Genre           | IMDB_Rating | Overview                          | Director             |
|---------------|---------------|-----------------|-------------|-----------------------------------|----------------------|
| Inception     | 2010          | Action, Sci-Fi  | 8.8         | A thief who steals corporate...   | Christopher Nolan    |

---
