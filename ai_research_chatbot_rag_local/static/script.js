const uploadForm = document.getElementById("upload-form");
const pdfInput = document.getElementById("pdf-file");
const fileName = document.getElementById("file-name");
const uploadStatus = document.getElementById("upload-status");
const uploadButton = document.getElementById("upload-button");
const questionForm = document.getElementById("question-form");
const questionInput = document.getElementById("question");
const chat = document.getElementById("chat");

let paperReady = false;

pdfInput.addEventListener("change", () => {
  fileName.textContent = pdfInput.files.length ? pdfInput.files[0].name : "Up to 20 MB · Text-based PDFs work best";
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!pdfInput.files.length) {
    setStatus("Choose a PDF file first.", "error");
    return;
  }

  const formData = new FormData();
  formData.append("pdf", pdfInput.files[0]);
  uploadButton.disabled = true;
  uploadButton.textContent = "Indexing... (may take a while)";
  setStatus("Extracting text and indexing the paper...", "");

  try {
    const response = await fetch("/upload", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok || !data.success) {
      paperReady = false;
      setStatus(data.message || "Could not load PDF.", "error");
      return;
    }
    paperReady = true;
    setStatus(data.message, "success");
    addMessage("bot", "Your paper is ready. Ask a question in your own words.");
  } catch {
    paperReady = false;
    setStatus("Could not connect to Flask. Check that app.py is running.", "error");
  } finally {
    uploadButton.disabled = false;
    uploadButton.textContent = "Upload and index paper";
  }
});

questionForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage("user", question);
  questionInput.value = "";

  if (!paperReady) {
    addMessage("bot", "Upload and index a research paper first.");
    return;
  }

  const loading = addMessage("bot", "Searching the paper and asking the hosted AI model...");
  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });
    const data = await response.json();
    loading.remove();

    if (!response.ok || !data.success) {
      addMessage("bot", data.message || "Something went wrong.");
      return;
    }
    const sourceText = data.sources && data.sources.length
      ? `Source pages: ${data.sources.join(", ")}`
      : "";
    addMessage("bot", data.answer, sourceText);
  } catch {
    loading.remove();
    addMessage("bot", "Could not connect to the website backend. Please try again.");
  }
});

document.querySelectorAll(".suggestions button").forEach((button) => {
  button.addEventListener("click", () => {
    questionInput.value = button.textContent;
    questionInput.focus();
  });
});

function setStatus(message, type) {
  uploadStatus.textContent = message;
  uploadStatus.className = "status" + (type ? ` ${type}` : "");
}

function addMessage(type, text, sourceText = "") {
  const row = document.createElement("div");
  row.className = `message ${type === "user" ? "user" : "bot"}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = type === "user" ? "YOU" : "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  if (sourceText) {
    const sources = document.createElement("span");
    sources.className = "sources";
    sources.textContent = sourceText;
    bubble.appendChild(sources);
  }

  row.append(avatar, bubble);
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;
  return row;
}
