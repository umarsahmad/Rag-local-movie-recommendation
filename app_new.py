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

# ---------- Build Prompt Template (Stricter) ------------
template = """
You are a helpful movie data assistant. You can only answer using the MOVIE ENTRIES below. 
If you don't find enough information in these entries, respond with "I don't know."

MOVIE ENTRIES:
{reviews}

QUESTION:
{question}

ANSWER (based only on the entries above):
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm

# ---------- Interactive Chat Loop ------------
print("\n🎬 Movie Review Chatbot Ready!")
print("Ask about movie names, ratings, genres, emotions, or anything about the data.")
print("Type 'q' to quit.\n")

while True:
    question = input("You: ").strip()
    if question.lower() in ("q", "quit", "exit"):
        print("Goodbye! 👋")
        break

    if not question:
        continue

    # Retrieve relevant entries
    print("🔍 Retrieving relevant entries from the dataset...")
    docs = retriever.invoke(question)

    if not docs:
        print("\n🤖 Assistant: I couldn't find anything in the dataset for that question.\n")
        continue

    reviews_text = "\n\n".join([doc.page_content for doc in docs])

    # Ask LLM
    print("💭 Thinking...")
    result = chain.invoke({"reviews": reviews_text, "question": question})

    print(f"\n🤖 Assistant: {result}\n")
