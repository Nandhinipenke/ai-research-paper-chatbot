import math
import os
import re
import time
from collections import Counter
import requests
from flask import Flask, render_template, request, jsonify
from pypdf import PdfReader

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash").strip()

document_chunks = []
document_vectors = []
document_idf = {}


def tokenize(text):
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


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
    global document_chunks, document_vectors, document_idf
    if not chunks:
        raise ValueError("No readable text was found in the PDF.")

    vectors = []
    document_frequency = Counter()
    for chunk in chunks:
        counts = Counter(tokenize(chunk["text"]))
        vectors.append(counts)
        document_frequency.update(counts.keys())

    total = max(len(vectors), 1)
    idf = {
        word: math.log(1 + total / (1 + frequency)) + 1
        for word, frequency in document_frequency.items()
    }
    document_chunks = chunks
    document_vectors = vectors
    document_idf = idf


def retrieve_relevant_chunks(question, top_k=4):
    if not document_chunks or not document_vectors:
        return []

    query = question
    lower_question = question.lower()
    expansions = {
        "methodology": "method approach proposed experimental setup procedure",
        "limitations": "limitations weakness constraints future work challenges",
        "objective": "objective aim purpose goal contribution",
        "results": "results findings evaluation performance outcome",
        "conclusion": "conclusion summary future work",
    }
    for key, terms in expansions.items():
        if key in lower_question:
            query += " " + terms

    query_terms = set(tokenize(query))
    if not query_terms:
        return []

    scored = []
    for index, counts in enumerate(document_vectors):
        score = sum(
            document_idf.get(term, 0.0) * (1 + math.log(counts[term]))
            for term in query_terms
            if counts.get(term, 0) > 0
        )
        if score > 0:
            scored.append((score, index))

    scored.sort(reverse=True)
    return [document_chunks[i] for _, i in scored[:top_k]]


def generate_answer(question, relevant_chunks):
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it in Render Environment."
        )
    if not relevant_chunks:
        return (
            "I could not find a clearly relevant passage in this paper. "
            "Try asking with different keywords, or check whether the PDF contains selectable text."
        )

    context = "\n\n".join(
        f"[Page {chunk['page']}] {chunk['text']}"
        for chunk in relevant_chunks
    )
    prompt = f"""
You are a helpful research-paper assistant.
Answer using only the supplied paper excerpts.
Use simple English and cite page numbers such as (Page 2).
If the excerpts do not contain the answer, say that the paper excerpts do not provide enough information. Do not invent facts.

Question: {question}

Paper excerpts:
{context}

Answer based only on these excerpts.
"""
    api_root = "https://generativelanguage.googleapis.com/v1beta"
    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 700},
    }

    # If the configured model is unavailable, discover the models exposed
    # to this API key and try supported candidates until one responds.
    response = _call_gemini_model(api_root, GEMINI_MODEL, headers, payload)
    if response.status_code == 404:
        try:
            model_response = requests.get(
                f"{api_root}/models",
                headers={"x-goog-api-key": GEMINI_API_KEY},
                timeout=(5, 15),
            )
            if model_response.status_code == 200:
                available = model_response.json().get("models", [])
                supported = [
                    item.get("name", "").removeprefix("models/")
                    for item in available
                    if "generateContent" in item.get("supportedGenerationMethods", [])
                ]
                preferred = [
                    "gemini-3.8-flash",
                    "gemini-3.8-flash-lite",
                    "gemini-3-flash",
                    "gemini-2.5-flash",
                    "gemini-2.5-flash-lite",
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-lite",
                ]
                ordered_models = []
                for name in preferred + supported:
                    if name and name not in ordered_models and name != GEMINI_MODEL:
                        ordered_models.append(name)

                for model_name in ordered_models:
                    candidate = _call_gemini_model(api_root, model_name, headers, payload)
                    response = candidate
                    if candidate.status_code != 404:
                        break
            else:
                print(
                    "Gemini model discovery failed:",
                    model_response.status_code,
                    model_response.text[:300],
                    flush=True,
                )
        except requests.exceptions.RequestException as error:
            print("Gemini model discovery error:", str(error), flush=True)

    if response is None:
        raise RuntimeError("Gemini did not return a response. Please try again.")
    if response.status_code in (401, 403):
        raise RuntimeError(
            "Gemini API key is invalid or access is denied. Check GEMINI_API_KEY "
            "in Render Environment and ensure the key is from Google AI Studio."
        )
    if response.status_code == 404:
        raise RuntimeError(
            "Google Gemini could not find a model available to this API key. "
            "Check the key in Google AI Studio and make sure the Generative Language API is enabled. "
            f"Google response: {response.text[:350]}"
        )
    if response.status_code == 429:
        raise RuntimeError("Gemini quota or rate limit reached. Check Google AI Studio usage limits.")
    if response.status_code != 200:
        raise RuntimeError(f"Gemini API error ({response.status_code}): {response.text[:400]}")

    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini returned no answer. Try another question.")
    parts = candidates[0].get("content", {}).get("parts", [])
    answer = "\\n".join(
        part.get("text", "") for part in parts if part.get("text")
    ).strip()
    if not answer:
        raise RuntimeError("Gemini returned an empty answer.")
    return answer


def _call_gemini_model(api_root, model_name, headers, payload):
    url = f"{api_root}/models/{model_name}:generateContent"
    for attempt in range(3):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=(5, 25),
            )
        except requests.exceptions.RequestException as error:
            if attempt == 2:
                raise RuntimeError(f"Could not connect to Gemini: {error}")
            time.sleep(2)
            continue

        if response.status_code not in (429, 500, 502, 503, 504):
            return response
        if attempt < 2:
            time.sleep(2)
    return response


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    pdf_file = request.files.get("pdf")
    if not pdf_file or not pdf_file.filename:
        return jsonify({"success": False, "message": "Choose a PDF file first."}), 400
    if not pdf_file.filename.lower().endswith(".pdf"):
        return jsonify({"success": False, "message": "Only PDF files are supported."}), 400

    try:
        chunks = read_pdf(pdf_file)
        if not chunks:
            return jsonify({
                "success": False,
                "message": "No selectable text found. Scanned PDFs need OCR before this app can read them.",
            }), 400
        index_document(chunks)
        pages = len({chunk["page"] for chunk in chunks})
        return jsonify({
            "success": True,
            "message": f"Paper loaded: {len(chunks)} text chunks across {pages} readable pages.",
            "chunks": len(chunks),
            "pages": pages,
        })
    except Exception as error:
        return jsonify({"success": False, "message": str(error)}), 400


@app.route("/ask", methods=["POST"])
def ask():
    if not document_chunks:
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
        print("Ask endpoint error:", str(error), flush=True)
        return jsonify({"success": False, "message": str(error)}), 502


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        "success": False,
        "message": "PDF is too large. Please upload a file smaller than 20 MB.",
    }), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
