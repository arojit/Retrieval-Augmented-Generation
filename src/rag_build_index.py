import os, json, faiss, numpy as np, tiktoken
from sentence_transformers import SentenceTransformer
from pathlib import Path
from pypdf import PdfReader
from pdfminer.high_level import extract_text as pm_extract_text

EMB = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
enc = tiktoken.get_encoding("cl100k_base")


def read_text_file(fp: Path) -> str:
    return fp.read_text(errors="ignore")

def read_pdf_file(fp: Path) -> str:
    # Primary: pypdf (fast, no external binaries)
    try:
        with open(fp, "rb") as f:
            reader = PdfReader(f)
            pages = [(p.extract_text() or "") for p in reader.pages]
        text = "\n".join(pages)
        if text.strip():
            return text
    except Exception:
        pass

    # Fallback: pdfminer.six (heavier, but often extracts more)
    try:
        return pm_extract_text(str(fp)) or ""
    except Exception:
        # If both fail (e.g., scanned PDFs), return empty string
        # (You can later add OCR with pytesseract if needed.)
        return ""

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
    suf = fp.suffix.lower()
    if suf in {".md", ".txt", ".html", ".pdf"}:
        print(f"-----file: {fp.name}")
        if suf == ".pdf":
            text = read_pdf_file(fp)
        else:
            text = read_text_file(fp)
        
        if not text.strip():
            continue

        for c in chunk(text):
            docs.append({"id": len(docs), "text": c, "source": str(fp)})

embs = EMB.encode([d["text"] for d in docs], normalize_embeddings=True, show_progress_bar=True)
index = faiss.IndexFlatIP(embs.shape[1])
index.add(np.array(embs, dtype=np.float32))

faiss.write_index(index, "data/rag/faiss.index")
Path("data/rag/segments.jsonl").write_text("\n".join(json.dumps(d) for d in docs))
print(f"Indexed {len(docs)} chunks.")
