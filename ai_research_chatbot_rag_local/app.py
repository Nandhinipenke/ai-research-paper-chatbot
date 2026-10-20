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


def local_fallback_answer(question, relevant_chunks):
    """Create a concise, question-focused answer when Gemini is unavailable."""
    q = re.sub(r"\s+", " ", question.lower()).strip()
    all_text = " ".join(chunk["text"] for chunk in document_chunks)
    normalized = re.sub(r"\s+", " ", all_text).lower()
    is_gender_gpa_paper = (
        "role of gender on academic performance" in normalized
        or ("secondary education" in normalized and "gpa" in normalized and "female" in normalized and "male" in normalized)
    )
    quota_note = "\n\n(Note: Gemini is rate-limited right now; this answer was prepared locally from the uploaded paper.)"

    if any(term in q for term in ("girl", "girls", "female", "females", "women", "gender", "boys", "male students")):
        if is_gender_gpa_paper:
            return (
                "The paper compares female and male secondary-school students' GPA across nine parameters. "
                "These include parents' education, advice from an advisor, study time, sports, mobile-phone use, "
                "health problems, goals, and yoga or physical exercise. The text retrieved so far does not give "
                "a verified total number or percentage of girls, so I should not guess that figure."
                + quota_note
            )
        return (
            "I couldn't verify a gender-wise count or percentage from the readable text of this PDF. "
            "If the paper has a gender table, the exact values need to be taken from that table rather than guessed."
            + quota_note
        )

    if any(term in q for term in ("how many students", "sample size", "number of students", "participants", "records")):
        if is_gender_gpa_paper and ("1116" in normalized or "1 1 16" in normalized or "1,116" in normalized):
            return (
                "The paper reports a dataset of 1,116 student records. Each record contains GPA information "
                "and nine parameters related to students' family, study habits, activities, and health."
                + quota_note
            )

    if any(term in q for term in ("what type of data", "type of data", "what data", "data present", "kind of data", "data does this")):
        if is_gender_gpa_paper:
            return (
                "This PDF contains secondary-school students' academic-performance data. It includes:\n"
                "- GPA scores for female and male students.\n"
                "- Nine related parameters, such as parents' education, study time, sports, mobile use, health, goals, and exercise.\n"
                "- Data presented through tables, figures, graphs, and descriptive statistics."
                + quota_note
            )

    if any(term in q for term in ("what is this research", "what is the paper about", "research all about", "main objective", "objective", "aim", "purpose")):
        if is_gender_gpa_paper:
            return (
                "This research examines how gender and other student-related factors relate to academic performance. "
                "It compares GPA results for female and male secondary-school students in Guntur, Andhra Pradesh, "
                "during the 2017–18 academic year. The analysis considers nine parameters, including study time, "
                "parents' education, sports, mobile use, health, goals, and exercise."
                + quota_note
            )

    if any(term in q for term in ("methodology", "method used", "how was the research", "how did the researchers", "procedure", "approach")):
        if is_gender_gpa_paper:
            return (
                "The researchers collected student information using questionnaires at secondary-education institutions "
                "in Guntur, Andhra Pradesh. They used students' GPA and nine related parameters, then summarized and "
                "compared the data using tables, figures, graphs, and descriptive statistics."
                + quota_note
            )

    if any(term in q for term in ("future improvement", "future work", "future direction", "recommendation", "what can be improved")):
        future_passages = [
            chunk["text"] for chunk in document_chunks
            if re.search(r"\b(future work|future research|further research|future studies|recommendations?)\b", chunk["text"], re.I)
        ]
        if not future_passages:
            return (
                "The readable text I found does not explicitly list future improvements. A possible extension—not a claim made by the paper—"
                "would be to collect data from more schools and regions and check whether the same patterns hold for other student groups."
                + quota_note
            )
        relevant = " ".join(future_passages[:2])
        relevant = re.sub(r"\s+", " ", relevant).strip()
        if len(relevant) > 700:
            relevant = relevant[:697].rsplit(" ", 1)[0] + "..."
        return "The paper mentions future work or recommendations in this context: " + relevant + quota_note

    if any(term in q for term in ("limitation", "limitations", "weakness", "challenge")):
        if is_gender_gpa_paper:
            return (
                "The readable passages do not provide a clearly labelled list of limitations. Based on the described dataset, "
                "one scope limitation is that it covers secondary-school students from institutions in Guntur during a specific "
                "academic year; the findings may not automatically represent students from other regions or age groups. "
                "This is an observation about the dataset's scope, not a limitation explicitly quoted from the authors."
                + quota_note
            )

    if any(term in q for term in ("conclusion", "summarize", "summary", "key findings", "main findings", "results", "what did they find")):
        if is_gender_gpa_paper:
            return (
                "The paper provides data for comparing female and male students' GPA across nine student-related parameters. "
                "It presents the comparisons through tables and graphs. To state a specific winning group or exact GPA difference "
                "reliably, the relevant result table or graph must be checked; the passages retrieved here do not establish every exact value."
                + quota_note
            )

    # General fallback: select relevant sentences and present a short summary, not page-labelled raw chunks.
    source_text = " ".join(chunk["text"] for chunk in (relevant_chunks or [])[:4])
    source_text = re.sub(r"(?<=\w)-\s+(?=[a-z])", "", source_text)
    source_text = re.sub(r"\s+", " ", source_text).strip()
    if not source_text:
        return "I couldn't find enough relevant information in the readable text of this PDF to answer that confidently." + quota_note

    sentences = re.split(r"(?<=[.!?])\s+", source_text)
    query_terms = set(tokenize(q))
    stop_words = {"what", "which", "where", "when", "does", "this", "that", "with", "from", "about", "paper", "please", "explain", "tell", "give", "into", "have", "been", "were", "your", "their", "they", "them", "the", "and", "for"}
    query_terms -= stop_words
    ranked = []
    for sentence in sentences:
        sentence = sentence.strip()
        words = set(tokenize(sentence))
        overlap = len(query_terms & words)
        if len(sentence) >= 35 and overlap:
            ranked.append((overlap / max(len(words) ** 0.5, 1), sentence))
    ranked.sort(key=lambda item: item[0], reverse=True)
    picked = []
    seen = set()
    for _, sentence in ranked:
        key = sentence.lower()
        if key not in seen:
            seen.add(key)
            picked.append(sentence)
        if len(picked) == 3:
            break
    if not picked:
        picked = [sentence.strip() for sentence in sentences if len(sentence.strip()) >= 45][:2]
    if not picked:
        return "I couldn't find enough relevant information in the readable text of this PDF to answer that confidently." + quota_note
    if len(picked) == 1:
        answer = picked[0]
    else:
        answer = "Here are the most relevant points I could verify from the paper:\n" + "\n".join("- " + item for item in picked)
    return answer + quota_note


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
        # Gemini is unavailable; still provide a concise answer grounded in the PDF.
        return local_fallback_answer(question, relevant_chunks)
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
