import os, json, faiss, numpy as np, tiktoken
from sentence_transformers import SentenceTransformer
from pathlib import Path

EMB = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
enc = tiktoken.get_encoding("cl100k_base")

def chunk(text, max_tokens=600, overlap=120):
    toks = enc.encode(text)
    out=[]
    i=0
    while i < len(toks):
        j = min(i+max_tokens, len(toks))
        out.append(enc.decode(toks[i:j]))
        i += max_tokens - overlap
    return out

docs=[]
for fp in Path("data/rag/raw").glob("**/*"):
    if fp.suffix.lower() in {".md",".txt",".html"}:
        print(f"-----file: {fp.name}")
        text = fp.read_text(errors="ignore")
        for c in chunk(text):
            docs.append({"id": len(docs), "text": c, "source": str(fp)})

embs = EMB.encode([d["text"] for d in docs], normalize_embeddings=True, show_progress_bar=True)
index = faiss.IndexFlatIP(embs.shape[1])
index.add(np.array(embs, dtype=np.float32))

faiss.write_index(index, "data/rag/faiss.index")
Path("data/rag/segments.jsonl").write_text("\n".join(json.dumps(d) for d in docs))
print(f"Indexed {len(docs)} chunks.")
