import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate import dispatch_model

# ---- CONFIG ----
MODEL_PATH = "./models/mistral-7b-4bit-32g"  # 👈 Your new local GPTQ model path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Using device: {DEVICE}")

# ---- LOAD TOKENIZER ----
print(f"📜 Loading tokenizer from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# ---- LOAD MODEL ----
print(f"🧠 Loading Mistral-7B GPTQ model from: {MODEL_PATH}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",            # Accelerate will try to use your GPU first
    torch_dtype=torch.float16,    # 4-bit quantized weights in fp16
    low_cpu_mem_usage=True
)

# ---- DISPATCH MODEL (Optional for small GPU) ----
print("🗂️  Dispatching model with offload_buffers=True (for safety)...")
model = dispatch_model(model, device_map="auto", offload_buffers=True)

# ---- CHECK DEVICE MAP ----
print(f"\n✅ Model device map:")
for module, dev in model.hf_device_map.items():
    print(f"   {module} -> {dev}")

# ---- CREATE PIPELINE ----
print("⚡ Creating generation pipeline...")
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

print(f"\n✅ Model fully loaded on {DEVICE}! Ready to chat.")
print("\n🟢 Mistral 7B (GPTQ) Chatbot - Local & GPU-powered")
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
