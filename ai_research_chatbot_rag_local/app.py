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

    # Discover models from Google's live model list. Never guess model IDs:
    # only try models actually returned for this API key with generateContent.
    response = _call_gemini_model(api_root, GEMINI_MODEL, headers, payload)
    if response.status_code in (400, 404, 503):
        try:
            model_response = requests.get(
                f"{api_root}/models",
                headers={"x-goog-api-key": GEMINI_API_KEY},
                timeout=(4, 10),
            )
            if model_response.status_code == 200:
                available = model_response.json().get("models", [])
                supported = []
                for item in available:
                    name = item.get("name", "").removeprefix("models/")
                    methods = item.get("supportedGenerationMethods", [])
                    lower_name = name.lower()
                    if (
                        name
                        and "generateContent" in methods
                        and not any(term in lower_name for term in (
                            "tts", "text-to-speech", "audio", "embedding",
                            "image", "live", "robotics"
                        ))
                    ):
                        supported.append(name)

                # Prefer fast models, but only if Google's live list confirms them.
                preferred_order = [
                    "gemini-3.8-flash",
                    "gemini-3.8-flash-lite",
                    "gemini-3-flash",
                    "gemini-3-flash-preview",
                    "gemini-2.5-flash",
                    "gemini-2.5-flash-lite",
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-lite",
                ]
                supported_set = set(supported)
                ordered_models = [
                    name for name in preferred_order
                    if name in supported_set and name != GEMINI_MODEL
                ]
                ordered_models += [
                    name for name in supported
                    if name not in ordered_models and name != GEMINI_MODEL
                ]

                # Try at most four models that the API explicitly lists.
                for model_name in ordered_models[:4]:
                    candidate = _call_gemini_model(api_root, model_name, headers, payload)
                    response = candidate
                    if candidate.status_code not in (400, 404, 503):
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
    if response.status_code in (429, 503):
        # If Gemini quota/capacity is unavailable, return the retrieved paper
        # passages rather than showing a backend error.
        excerpts = []
        for chunk in relevant_chunks[:3]:
            excerpt = re.sub(r"\s+", " ", chunk["text"]).strip()
            if len(excerpt) > 650:
                excerpt = excerpt[:647].rsplit(" ", 1)[0] + "..."
            excerpts.append(f"[Page {chunk['page']}] {excerpt}")
        if response.status_code == 429:
            opening = (
                "Gemini's API quota/rate limit is currently reached, so I cannot "
                "generate a rewritten answer right now. Here are the most relevant "
                "passages I found in your paper:"
            )
            closing = (
                "Please verify these passages in the original PDF. "
                "Check your project's current limits in Google AI Studio."
            )
        else:
            opening = (
                "Gemini is temporarily busy, so here are the most relevant "
                "paper passages instead of a generated summary:"
            )
            closing = "Please verify these passages in the original PDF and try again later."
        q = re.sub(r"\\s+", " ", question.lower()).strip()
        quota_note = " (Gemini is rate-limited, so this is a local fallback answer, not a newly generated AI response.)"
        if any(term in q for term in ("methodology", "method", "workflow", "how does it work")):
            return (
                "The methodology is: (1) upload the PDF, (2) extract text from each page, "
                "(3) clean and split the text into overlapping chunks, (4) retrieve the chunks "
                "most relevant to the question, (5) send those passages to the language model, "
                "and (6) return an answer with page references. "
                "In short: PDF → text extraction → chunking → retrieval → AI answer." + quota_note
            )
        if any(term in q for term in ("main objective", "objective", "aim of", "purpose of")):
            return (
                "The main objective is to help users understand and search research papers more easily "
                "by letting them ask questions about a PDF. The chatbot retrieves relevant passages "
                "and uses them as evidence for answers, with page references where possible." + quota_note
            )
        if ("use" in q and any(term in q for term in ("concept", "concent", "this", "it"))) or "applications" in q:
            return (
                "The use of this concept is to make long research papers easier to understand. "
                "A user can upload a PDF and ask questions instead of searching every page manually. "
                "The chatbot finds related passages and uses them to help explain the paper, locate "
                "methods or limitations, and identify key information. Answers should still be checked "
                "against the original paper." + quota_note
            )
        if any(term in q for term in ("limitation", "weakness", "challenge")):
            return (
                "The main limitations are: (1) scanned PDFs without selectable text may not be readable, "
                "(2) answers depend on whether the retrieval step finds the right passages, "
                "(3) the language model can still make unsupported claims, and "
                "(4) external AI quota or service outages can prevent generated answers. "
                "Check important details in the original paper." + quota_note
            )
        return opening + "\\n\\n" + "\\n\\n".join(excerpts) + "\\n\\n" + closing
    if response.status_code != 200:
        raise RuntimeError(f"Gemini API error ({response.status_code}): {response.text[:400]}")

    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini returned no answer. Try another question.")
    parts = candidates[0].get("content", {}).get("parts", [])
    answer = "\n".join(
        part.get("text", "") for part in parts if part.get("text")
    ).strip()
    if not answer:
        raise RuntimeError("Gemini returned an empty answer.")
    return answer


def _call_gemini_model(api_root, model_name, headers, payload):
    url = f"{api_root}/models/{model_name}:generateContent"
    # Keep each request short so a slow upstream model cannot cause the
    # Render/Gunicorn worker to be killed. At most one retry per model.
    for attempt in range(2):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=(4, 12),
            )
        except requests.exceptions.RequestException as error:
            if attempt == 1:
                raise RuntimeError(
                    "Gemini took too long to respond or could not be reached. "
                    "Please retry in a moment."
                ) from error
            time.sleep(1)
            continue

        if response.status_code not in (429, 500, 502, 503, 504):
            return response
        if attempt == 0:
            time.sleep(1)
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
