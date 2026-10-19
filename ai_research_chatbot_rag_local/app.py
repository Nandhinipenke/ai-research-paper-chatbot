import os
import re
import requests
from flask import Flask, render_template, request, jsonify
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"
# Change this model only to one currently available to your Hugging Face account/provider.
HF_MODEL = os.environ.get("HF_MODEL", "openai/gpt-oss-120b:fastest")
document_chunks = []
vectorizer = None
chunk_matrix = None


def split_into_chunks(text, page_number, chunk_size=1000, overlap=150):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(". ", start + chunk_size // 2, end)
            if boundary != -1:
                end = boundary + 1
        part = text[start:end].strip()
        if part:
            chunks.append({"text": part, "page": page_number})
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def read_pdf(file_storage):
    reader = PdfReader(file_storage.stream)
    chunks = []
    for page_number, page in enumerate(reader.pages, start=1):
        chunks.extend(split_into_chunks(page.extract_text() or "", page_number))
    return chunks


def index_document(chunks):
    global vectorizer, chunk_matrix
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    chunk_matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])
    return chunks


def retrieve_relevant_chunks(question, top_k=4):
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, chunk_matrix).ravel()
    indexes = scores.argsort()[::-1][:top_k]
    return [document_chunks[i] for i in indexes if scores[i] > 0]


def generate_answer(question, relevant_chunks):
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not configured. Add your Hugging Face token in Render → Environment, then redeploy.")
    if not relevant_chunks:
        return "I could not find a clearly relevant passage in this paper. Try asking with different keywords."
    context = "\n\n".join(f"[Page {c['page']}] {c['text']}" for c in relevant_chunks)
    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful research-paper assistant. Answer using only the supplied paper excerpts. Do not invent facts. If the excerpts do not contain enough information, say so. Use simple, clear English and cite page numbers such as (Page 2)."},
            {"role": "user", "content": f"Question: {question}\n\nPaper excerpts:\n{context}\n\nAnswer based only on these excerpts."}
        ],
        "temperature": 0.2,
        "stream": False,
        "max_tokens": 700
    }
    try:
        response = requests.post(
            HF_CHAT_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
            json=payload,
            timeout=120
        )
        if response.status_code in (401, 403):
            raise RuntimeError("Hugging Face token is invalid or does not have Inference Providers permission. Check HF_TOKEN in Render Environment.")
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.Timeout:
        raise RuntimeError("The hosted AI model timed out. Try again or choose an available model in HF_MODEL.")
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = response.text[:400]
        except Exception:
            pass
        raise RuntimeError(f"Hosted AI request failed ({response.status_code}): {detail or str(exc)}")
    except (requests.exceptions.RequestException, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Could not get a response from the hosted AI service: {exc}")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global document_chunks
    pdf_file = request.files.get("pdf")
    if not pdf_file or not pdf_file.filename:
        return jsonify({"success": False, "message": "Choose a PDF file first."}), 400
    if not pdf_file.filename.lower().endswith(".pdf"):
        return jsonify({"success": False, "message": "Only PDF files are supported."}), 400
    try:
        chunks = read_pdf(pdf_file)
        if not chunks:
            return jsonify({"success": False, "message": "No selectable text found. Scanned PDFs need OCR before this app can read them."}), 400
        index_document(chunks)
        document_chunks = chunks
        pages = len({chunk["page"] for chunk in chunks})
        return jsonify({"success": True, "message": f"Paper loaded: {len(chunks)} text chunks across {pages} readable pages.", "chunks": len(chunks), "pages": pages})
    except Exception as error:
        return jsonify({"success": False, "message": str(error)}), 400


@app.route("/ask", methods=["POST"])
def ask():
    if not document_chunks or vectorizer is None or chunk_matrix is None:
        return jsonify({"success": False, "message": "Upload a research paper first."}), 400
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"success": False, "message": "Type a question first."}), 400
    try:
        selected = retrieve_relevant_chunks(question)
        answer = generate_answer(question, selected)
        sources = list(dict.fromkeys(chunk["page"] for chunk in selected))
        return jsonify({"success": True, "answer": answer, "sources": sources})
    except Exception as error:
        return jsonify({"success": False, "message": str(error)}), 502


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"success": False, "message": "PDF is too large. Please upload a file smaller than 20 MB."}), 413


if __name__ == "__main__":
    # Local development: set HF_TOKEN in your terminal before running the app.
    app.run(debug=True)
