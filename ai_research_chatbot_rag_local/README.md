# AI Research Paper Chatbot — Hosted Gemini AI

A Flask + HTML/CSS/JavaScript research-paper chatbot. It extracts text from a PDF, retrieves relevant passages using TF-IDF, and asks the Google Gemini API to answer using those passages with page references.

## API key setup

The hosted app requires a Google Gemini API key. Create/manage it in Google AI Studio, then add it in **Render → your service → Environment** as `GEMINI_API_KEY`. Never commit the real key to GitHub or place it in browser-side JavaScript.

The default model is `gemini-2.5-flash`. If needed, set `GEMINI_MODEL` to a model available to your API key.

## Local setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
$env:GEMINI_API_KEY="paste_your_key_here"
python app.py
```

Open http://127.0.0.1:5000

## Deploy to Render

The Flask app is inside `ai_research_chatbot_rag_local`.

1. Create or open a Render Web Service connected to this GitHub repository.
2. Set **Root Directory** to `ai_research_chatbot_rag_local`.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120`
5. Add `GEMINI_API_KEY` in the service's Environment settings. Optionally set `GEMINI_MODEL` to `gemini-2.5-flash`.
6. Deploy and open the HTTPS URL Render provides.

Never commit the real API key. If a key is accidentally exposed, revoke it and create a new one.

## Notes / demo limitations

- The uploaded paper and its index are held in process memory; a service restart clears them. This simple demo is not designed for multiple users to upload papers concurrently.
- Free hosting may sleep and has resource/time limits. Gemini usage is subject to your account's quota and limits.
- Scanned/image-only PDFs need OCR, which is not included.
- Answers can be wrong; verify important claims against the cited page numbers.
- PDF upload limit: 20 MB.
