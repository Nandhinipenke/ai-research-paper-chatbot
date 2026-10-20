import os
import re
import requests
from flask import Flask, render_template, request, jsonify
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()

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
        chunks.extend(
            split_into_chunks(page.extract_text() or "", page_number)
        )

    return chunks


def index_document(chunks):
    global vectorizer, chunk_matrix

    if not chunks:
        raise ValueError("No readable text was found in the PDF.")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )
    chunk_matrix = vectorizer.fit_transform(
        [chunk["text"] for chunk in chunks]
    )


def retrieve_relevant_chunks(question, top_k=4):
    if vectorizer is None or chunk_matrix is None:
        return []

    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(
        question_vector, chunk_matrix
    ).ravel()

    indexes = scores.argsort()[::-1][:top_k]

    return [
        document_chunks[i]
        for i in indexes
        if scores[i] > 0
    ]


def generate_answer(question, relevant_chunks):
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it in Render Environment."
        )

    if not relevant_chunks:
        return (
            "I could not find a clearly relevant passage in this paper. "
            "Try asking with different keywords."
        )

    context = "\n\n".join(
        f"[Page {chunk['page']}] {chunk['text']}"
        for chunk in relevant_chunks
    )

    prompt = f"""
You are a helpful research-paper assistant.
Answer using only the supplied paper excerpts.
Do not invent facts. If the excerpts do not contain enough information,
say so. Use simple English and cite page numbers such as (Page 2).

Question: {question}

Paper excerpts:
{context}

Answer based only on these excerpts.
"""

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 700
        }
    }

    try:
        response = requests.post(
            url,
            headers={
                "x-goog-api-key": GEMINI_API_KEY,
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=120
        )

        if response.status_code in (401, 403):
            raise RuntimeError(
                "Gemini API key is invalid or access is denied. "
                "Check GEMINI_API_KEY in Render Environment."
            )

        if response.status_code == 429:
            raise RuntimeError(
                "Gemini quota or rate limit reached. Check your "
                "Google AI Studio usage limits and try again later."
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Gemini API error ({response.status_code}): "
                f"{response.text[:500]}"
            )

        data = response.json()
        candidates = data.get("candidates", [])

        if not candidates:
            raise RuntimeError(
                "Gemini returned no answer. Try another question."
            )

        parts = candidates[0].get("content", {}).get("parts", [])
        answer = "\n".join(
            part.get("text", "")
            for part in parts
            if part.get("text")
        ).strip()

        if not answer:
            raise RuntimeError("Gemini returned an empty answer.")

        return answer

    except requests.exceptions.Timeout:
        raise RuntimeError("Gemini timed out. Please try again.")
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Could not connect to Gemini: {exc}")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global document_chunks

    pdf_file = request.files.get("pdf")

    if not pdf_file or not pdf_file.filename:
        return jsonify({
            "success": False,
            "message": "Choose a PDF file first."
        }), 400

    if not pdf_file.filename.lower().endswith(".pdf"):
        return jsonify({
            "success": False,
            "message": "Only PDF files are supported."
        }), 400

    try:
        chunks = read_pdf(pdf_file)

        if not chunks:
            return jsonify({
                "success": False,
                "message": (
                    "No selectable text found. Scanned PDFs need OCR "
                    "before this app can read them."
                )
            }), 400

        index_document(chunks)
        document_chunks = chunks

        pages = len({chunk["page"] for chunk in chunks})

        return jsonify({
            "success": True,
            "message": (
                f"Paper loaded: {len(chunks)} text chunks across "
                f"{pages} readable pages."
            ),
            "chunks": len(chunks),
            "pages": pages
        })

    except Exception as error:
        return jsonify({
            "success": False,
            "message": str(error)
        }), 400


@app.route("/ask", methods=["POST"])
def ask():
    if not document_chunks or vectorizer is None or chunk_matrix is None:
        return jsonify({
            "success": False,
            "message": "Upload a research paper first."
        }), 400

    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()

    if not question:
        return jsonify({
            "success": False,
            "message": "Type a question first."
        }), 400

    try:
        selected = retrieve_relevant_chunks(question)
        answer = generate_answer(question, selected)

        sources = list(dict.fromkeys(
            chunk["page"] for chunk in selected
        ))

        return jsonify({
            "success": True,
            "answer": answer,
            "sources": sources
        })

    except Exception as error:
        return jsonify({
            "success": False,
            "message": str(error)
        }), 502


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        "success": False,
        "message": (
            "PDF is too large. Please upload a file smaller than 20 MB."
        )
    }), 413


if __name__ == "__main__":
    app.run(debug=True)
