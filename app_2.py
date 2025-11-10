import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------- Load Vector Store ------------
print("📦 Loading vector store...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = Chroma(
    collection_name="movie_reviews",
    persist_directory="./movie_reviews_db",
    embedding_function=embeddings
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# ---------- Load TinyLlama ------------
print("🧠 Loading TinyLlama...")
MODEL_PATH = "./models/tiny-llama-1b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map=DEVICE,
    torch_dtype=torch.float16
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.2,
    top_p=0.9
)

llm = HuggingFacePipeline(pipeline=pipe)

# ---------- Define Prompt Templates ------------
review_template = """
You are a movie expert. ONLY use the entries below to answer.
If the movie is not found in these entries, respond with "I don't know."

MOVIE ENTRIES:
{reviews}

QUESTION:
{question}

ANSWER (summarize the movie using Title, Year, Genre, IMDB_Rating, Overview, and Director from the entries):
"""

rating_template = """
You are a movie database assistant. ONLY use the entries below to answer.
List movie titles with IMDB ratings around the number the user asked for.
If no matching movies are found, say "I don't know."

MOVIE ENTRIES:
{reviews}

QUESTION:
{question}

ANSWER (list only matching Titles, no extra explanation):
"""

genre_template = """
You are a movie database assistant. ONLY use the entries below to answer.
List movie titles only for the genres the user asked for.
If no matching movies are found, say "I don't know."

MOVIE ENTRIES:
{reviews}

QUESTION:
{question}

ANSWER (list only matching Titles, no extra explanation):
"""

director_template = """
You are a movie database assistant. ONLY use the entries below to answer.
List movie titles directed by the director the user asked for.
If no matching movies are found, say "I don't know."

MOVIE ENTRIES:
{reviews}

QUESTION:
{question}

ANSWER (list only matching Titles, no extra explanation):
"""

general_template = """
You are a helpful movie data assistant. ONLY use the entries below to answer.
If you cannot find the answer in these entries, say "I don't know."

MOVIE ENTRIES:
{reviews}

QUESTION:
{question}

ANSWER (based only on the entries above):
"""

# ---------- Interactive Chat Loop ------------
print("\n🎬 Movie Chatbot Ready!")
print("Examples you can try:")
print("- give me movie review for Inception")
print("- give me movie titles with IMDB ratings 8")
print("- give me movie titles with genres 'Drama','Thriller'")
print("- give me movies directed by Christopher Nolan")
print("Type 'q' to quit.\n")

while True:
    question = input("You: ").strip()
    if question.lower() in ("q", "quit", "exit"):
        print("Goodbye! 👋")
        break

    if not question:
        continue

    # 🔎 Classify the question type
    q_lower = question.lower()
    if q_lower.startswith("give me movie review"):
        selected_template = review_template
    elif "ratings" in q_lower or "imdb" in q_lower:
        selected_template = rating_template
    elif "genres" in q_lower or "genre" in q_lower:
        selected_template = genre_template
    elif "director" in q_lower:
        selected_template = director_template
    else:
        selected_template = general_template

    # 🔍 Retrieve relevant entries
    print("🔍 Retrieving relevant entries from the dataset...")
    docs = retriever.invoke(question)

    if not docs:
        print("\n🤖 Assistant: I couldn't find anything in the dataset for that question.\n")
        continue

    reviews_text = "\n\n".join([doc.page_content for doc in docs])

    # 📝 Create the chain with selected template
    prompt = ChatPromptTemplate.from_template(selected_template)
    chain = prompt | llm

    # 🤖 Ask LLM
    print("💭 Thinking...")
    result = chain.invoke({"reviews": reviews_text, "question": question})

    # ✅ Clean up output to avoid "Assistant:" rambling
    if "Assistant:" in result:
        result = result.split("Assistant:")[-1].strip()
    result = result.strip()

    print(f"\n🤖 Assistant: {result}\n")
