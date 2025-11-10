import os
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 1️⃣ Load your CSV
df = pd.read_csv("Movies_Reviews_modified_version1.csv")

# Preprocessing:
df.drop(columns=['Resenhas', 'Unnamed: 0', 'Description'], inplace=True)
print(df.movie_name.nunique())
df.drop_duplicates(subset='movie_name', keep='first', inplace=True)
print(df['movie_name'].value_counts())


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
        Movie: {row['movie_name']}
        Review: {row['Reviews']}
        Genres: {row['genres']}
        Rating: {row['Ratings']}
        Emotion: {row['emotion']}
        """
        doc = Document(
            page_content=text.strip(),
            metadata={"movie_name": row["movie_name"]},
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
