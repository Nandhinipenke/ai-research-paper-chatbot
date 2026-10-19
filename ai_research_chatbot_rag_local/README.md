# AI Research Paper Chatbot — Hosted AI Website

Flask + HTML/CSS/JavaScript research-paper chatbot. It extracts text from a PDF, retrieves relevant passages using TF-IDF, and asks a hosted Hugging Face Inference Provider model to draft an answer grounded in those passages.

## Important: API token required

This hosted version requires a Hugging Face access token with **Make calls to Inference Providers** permission. Create one in Hugging Face Settings → Access Tokens. Keep it secret. Never paste it into `app.py`, HTML/JavaScript, or GitHub.

For local Windows testing in PowerShell, set the token for the current terminal session:

```powershell
$env:HF_TOKEN="paste_your_token_here"
python app.py
```

Do not commit the token. The app defaults to model `openai/gpt-oss-120b:fastest`; provider availability, rate limits, and cost depend on your Hugging Face account and selected provider. If needed, set `HF_MODEL` to a model currently available to your account.

## Local setup (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
$env:HF_TOKEN="paste_your_token_here"
python app.py
```

Open http://127.0.0.1:5000

## Deploy to Render

1. Push this folder's contents to a GitHub repository.
2. In Render, create **New → Web Service** and connect the repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. In the Render service's **Environment** settings, add `HF_TOKEN` with your secret token as the value. Optionally add `HF_MODEL` if using a different supported model.
6. Deploy. Render gives the service an HTTPS `onrender.com` URL.

Never add the real token to GitHub. If it is accidentally committed, revoke it and create a new one.

## Notes / demo limitations

- Uploaded paper and index are stored in process memory. They are temporary and may be lost when the service restarts or sleeps. This starter version is designed for a simple student demo, not multiple users at once.
- Free hosting can sleep and has resource/time limits. AI provider usage can have separate limits or charges.
- Scanned/image-only PDFs need OCR, which is not included.
- Answers can be wrong; verify important claims against the cited page numbers.
- The PDF size limit is 20 MB.
