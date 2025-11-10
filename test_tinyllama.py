import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---- CONFIG ----
MODEL_PATH = "./models/tiny-llama-1b"  # 👈 Put your downloaded TinyLlama folder path here
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Using device: {DEVICE}")

# ---- LOAD TOKENIZER ----
print(f"📜 Loading tokenizer from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# ---- LOAD MODEL ----
print(f"🧠 Loading TinyLlama model from: {MODEL_PATH}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map=DEVICE,           # load entire model on GPU if possible
    torch_dtype=torch.float16,   # efficient fp16 on CUDA
)

# ---- CREATE PIPELINE ----
print("⚡ Creating text generation pipeline...")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # device=0 if DEVICE == "cuda" else -1,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
    top_p=0.95
)

print(f"\n✅ Model fully loaded on {DEVICE}! Ready to chat.")
print("\n🟢 TinyLlama Chatbot (local, GPU-powered if available)")
print("Type 'exit' or 'quit' to stop.\n")

# ---- CHAT LOOP ----
chat_history = []

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("exit", "quit"):
        print("Goodbye! 👋")
        break

    if not user_input:
        continue

    # Add user message to history
    chat_history.append(f"User: {user_input}")

    # Build prompt with history
    prompt_text = "\n".join(chat_history) + "\nAssistant:"

    # Generate
    print("💭 Generating answer (please wait)...")
    output = pipe(prompt_text)[0]['generated_text']

    # Extract only new assistant reply
    assistant_reply = output[len(prompt_text):].strip()
    print(f"Assistant: {assistant_reply}\n")

    # Add assistant reply to history
    chat_history.append(f"Assistant: {assistant_reply}")
