import os
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 1️⃣ Load your CSV
df = pd.read_csv("imdb_top_1000.csv")


# Preprocess the dataset:

df.drop(columns=['Poster_Link', 'Certificate', 'Runtime', 'Meta_score', 'Star1', 'Star2', 'Star3', 'Star4', 'No_of_Votes', 'Gross'], inplace=True)
df.drop_duplicates(subset='Series_Title', keep='first', inplace=True)
print(len(df))


# ✅ Inspect
print(df.head())

# 2️⃣ Define embeddings model
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

db_location = "./movie_reviews_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    print("🛠️ Creating new vector store...")
    documents = []
    ids = []

    for i, row in df.iterrows():
        text = f"""
        Title: {row['Series_Title']}
        Year: {row['Released_Year']}
        Genre: {row['Genre']}
        IMDB_Rating: {row['IMDB_Rating']}
        Overview: {row['Overview']}
        Director: {row['Director']}
        """
        doc = Document(
            page_content=text.strip(),
            metadata={"title": row["Series_Title"]},
            id=str(i)
        )
        documents.append(doc)
        ids.append(str(i))

# 3️⃣ Initialize DB
vector_store = Chroma(
    collection_name="movie_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

# 4️⃣ Add docs if needed
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    print("✅ Vector store built and saved!")
else:
    print("✅ Vector store already exists, skipping.")
