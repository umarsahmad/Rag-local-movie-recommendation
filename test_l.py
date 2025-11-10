import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate import dispatch_model

# ---- CONFIG ----
MODEL_PATH = "./models/llama-7b-4bit-128g"  # 👈 Your local GPTQ model path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- LOAD TOKENIZER ----
print(f"📜 Loading tokenizer from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, legacy=True)

# ---- LOAD MODEL ----
print(f"🧠 Loading model from: {MODEL_PATH}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",              # Hugging Face Accelerate handles GPU/CPU
    torch_dtype=torch.float16,      # FP16 for GPU efficiency
    low_cpu_mem_usage=True
)

# Optional but recommended on small GPUs: ensure offload_buffers is used
print("🗂️  Dispatching model with offload_buffers=True for small GPU...")
model = dispatch_model(model, device_map="auto", offload_buffers=True)

# ---- CHECK DEVICES ----
print(f"\n✅ Model device map:")
for module, device in model.hf_device_map.items():
    print(f"   {module} -> {device}")

# ---- CREATE PIPELINE ----
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if DEVICE == "cuda" else -1,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
    top_p=0.95
)

print(f"\n✅ Model fully loaded on {DEVICE} (with offloading if needed)!")
print("\n🟢 LLaMA 2 Chatbot (local, GPU/CPU-aware)")
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
    print("💭 Generating answer...")
    output = pipe(prompt_text)[0]['generated_text']

    # Extract only new assistant reply
    assistant_reply = output[len(prompt_text):].strip()
    print(f"Assistant: {assistant_reply}\n")

    # Add assistant reply to history
    chat_history.append(f"Assistant: {assistant_reply}")
