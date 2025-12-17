import os
from flask import Flask, request, render_template_string
from dotenv import load_dotenv
from summarizer import extract_text_from_pdf, refine_summarize_text

load_dotenv()  # loads GOOGLE_API_KEY from .env if present
# For langchain-google-genai, env var name should be GOOGLE_API_KEY
assert os.getenv("GOOGLE_API_KEY"), "Set GOOGLE_API_KEY in environment!"

app = Flask(__name__)

PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Refine Summarizer</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 920px; margin: 40px auto; padding: 0 16px; }
    textarea { width: 100%; min-height: 180px; }
    .card { border: 1px solid #eee; padding: 16px; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,.04); }
    .row { display: grid; gap: 16px; }
    .btn { background: black; color: white; padding: 10px 16px; border-radius: 10px; border: 0; cursor: pointer; }
    .btn:disabled { opacity: .6; cursor: not-allowed; }
    label { font-weight: 600; }
    pre { white-space: pre-wrap; }
  </style>
</head>
<body>
  <h1>Refine Summarizer (Gemini + LangChain)</h1>
  <p>Paste text or upload a PDF. Uses RefineChain to build a high-quality summary.</p>

  <form class="card" method="POST" enctype="multipart/form-data">
    <div class="row">
      <div>
        <label>Paste text</label>
        <textarea name="text" placeholder="Paste text here...">{{ text or '' }}</textarea>
      </div>
      <div>
        <label>Or upload PDF</label><br/>
        <input type="file" name="pdf" accept="application/pdf" />
      </div>
      <div>
        <label>Model</label><br/>
        <select name="model">
          <option value="gemini-2.0-flash-exp" {% if model=='gemini-2.0-flash-exp' %}selected{% endif %}>gemini-2.0-flash-exp (fast, free)</option>
          <option value="gemini-1.5-pro" {% if model=='gemini-1.5-pro' %}selected{% endif %}>gemini-1.5-pro (richer)</option>
        </select>
      </div>
      <div>
        <label>Chunk size</label><br/>
        <input type="number" name="chunk_size" value="{{ chunk_size or 1800 }}" min="500" max="6000"/>
      </div>
      <div>
        <label>Chunk overlap</label><br/>
        <input type="number" name="chunk_overlap" value="{{ chunk_overlap or 200 }}" min="0" max="1000"/>
      </div>
      <div>
        <button class="btn" type="submit">Summarize</button>
      </div>
    </div>
  </form>

  {% if summary %}
  <div class="card">
    <h2>Summary</h2>
    <pre>{{ summary }}</pre>
  </div>
  {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    summary = None
    model = "gemini-2.0-flash-exp"
    chunk_size = 1800
    chunk_overlap = 200

    if request.method == "POST":
        model = request.form.get("model", model)
        try:
            chunk_size = int(request.form.get("chunk_size", chunk_size))
            chunk_overlap = int(request.form.get("chunk_overlap", chunk_overlap))
        except:
            pass

        # Prefer pasted text; if empty, try the PDF
        text = (request.form.get("text") or "").strip()
        if not text and "pdf" in request.files and request.files["pdf"].filename:
            pdf_file = request.files["pdf"]
            text = extract_text_from_pdf(pdf_file)

        if text:
            try:
                summary = refine_summarize_text(
                    text=text,
                    model=model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            except Exception as e:
                summary = f"Error while summarizing: {e}"
        else:
            summary = "Please paste text or upload a PDF."

    return render_template_string(PAGE,
        text=text, summary=summary, model=model,
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

if __name__ == "__main__":
    # For local dev: python app.py
    app.run(host="0.0.0.0", port=5000, debug=True)