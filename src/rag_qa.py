import os, json, faiss, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

# ----- Auth (env var or prior CLI login) -----
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", None)
if HF_TOKEN:
    login(HF_TOKEN)

# ----- Models -----
# v0.3 is gated; v0.2 is public
# MODEL_ID = os.getenv("HF_LM_ID", "mistralai/Mistral-7B-Instruct-v0.2")
MODEL_ID = os.getenv("HF_LM_ID", "Qwen/Qwen2.5-3B-Instruct")
EMB = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ----- FAISS + Segments -----
INDEX_PATH = "data/rag/faiss.index"
SEGS_PATH = "data/rag/segments.jsonl"

index = faiss.read_index(INDEX_PATH)
segs = [json.loads(l) for l in open(SEGS_PATH, "r", encoding="utf-8")]

# Choose device/dtype safely on Mac
if torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16         # MPS wants fp16; bf16 is unstable
elif torch.cuda.is_available():
    device = "cuda"
    # bf16 if Ampere+ (capability >= 8), else fp16
    major, _ = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if major >= 8 else torch.float16
else:
    device = "cpu"
    dtype = torch.float32

# ----- Tokenizer/Model -----
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

lm = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map={"": device},
    low_cpu_mem_usage=True,
    attn_implementation="eager"    # safer on MPS than sdpa/flash
)

gen = pipeline(
    "text-generation",
    model=lm,
    tokenizer=tok,
    # important so output doesn't include the prompt
    return_full_text=False
)

def retrieve(q, k=5):
    # If your FAISS index is IP (recommended with normalized vectors),
    # normalize_embeddings=True is correct. If the index is L2, remove normalization.
    qv = EMB.encode([q], normalize_embeddings=True)
    D, I = index.search(np.array(qv, dtype=np.float32), k)
    return [segs[i] for i in I[0]]

def build_messages(question, contexts):
    # Build a chat prompt using Mistral’s chat template
    # Expect each ctx to have 'text' and optionally 'source'
    ctx_text = "\n\n---\n\n".join(
        (c["text"] if isinstance(c, dict) and "text" in c else str(c))
        for c in contexts
    )
    system_msg = (
        "You are a helpful RAG assistant. Use the provided Context to answer the user's question.\n"
        "Cite sources in brackets like [1], [2] referencing the items in the Context order when possible.\n"
        "If the answer is not in the context, say you don't know."
    )
    user_msg = f"Context:\n{ctx_text}\n\nQuestion: {question}\nAnswer with references."

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

def answer(q, k=5, max_new_tokens=400):
    print(F"Query: {q}")
    ctx = retrieve(q, k=k)

    # Build a simple reference map for citations like [1], [2], ...
    refs = []
    for i, c in enumerate(ctx, start=1):
        src = None
        if isinstance(c, dict):
            src = c.get("source") or c.get("metadata") or c.get("id")
        refs.append((i, src))

    messages = build_messages(q, ctx)
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    out = gen(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.05
    )[0]["generated_text"]

    # Append a simple “Sources” footer if you have them
    sources_block = ""
    cleaned_refs = [f"[{i}] {s}" for (i, s) in refs if s]
    if cleaned_refs:
        sources_block = "\n\nSources:\n" + "\n".join(cleaned_refs)

    return out + sources_block

if __name__ == "__main__":
    print("--------------------------------------------------------")
    print(answer("What's our refund policy?"))
    print("========================================================")

    print("--------------------------------------------------------")
    print(answer("How do I change my email"))
    print("========================================================")

    print("--------------------------------------------------------")
    print(answer("When can I get full refund?"))
    print("========================================================")

    print("--------------------------------------------------------")
    print(answer("Address of Sherlock Holmes"))
    print("========================================================")
