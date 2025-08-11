# 2023 Honda CR-V Assistant

A lightweight, local **RAG** (retrieve-and-generate) console app that answers questions about the **2023 Honda CR-V owner’s manual**.
It uses **Azure Form Recognizer** to extract text from a PDF, **Azure OpenAI** to embed and chat, and **cosine similarity** to retrieve the most relevant chunks.

---

## What this script does

1. **Extracts text** from `2023-crv.pdf` using Form Recognizer’s `prebuilt-read`.
2. **Chunks** the text into \~500-char segments.
3. **Embeds** each chunk with Azure OpenAI (`text-embedding-3-large`) and caches results to `crv_embeddings.pkl`.
4. Runs a **simple RAG loop**: for each user query, finds the most similar chunks and calls a chat model (`o4-mini`) to generate an answer.
5. Exposes a **CLI**: type your question, get an answer, type `exit` to quit.

---

## Prerequisites

* **Python** 3.9–3.12
* An **Azure account** with:

  * **Azure AI Document Intelligence / Form Recognizer** (endpoint + key)
  * **Azure OpenAI** (endpoint + key) and **deployed models**:

    * An **embeddings** deployment named `text-embedding-3-large` (or update the script)
    * A **chat** deployment named `o4-mini` (or update the script)
* The **2023 Honda CR-V manual** PDF saved as `2023-crv.pdf` in the repo root
  (or update `PDF_PATH` in the script)

---

## Installation

```bash
# create and activate a virtual environment (recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# install dependencies
pip install azure-ai-formrecognizer
pip install azure-core
pip install scikit-learn
pip install openai
```

> `pickle` and `os` are from Python’s standard library (no install needed).

---

## Configuration

Update these placeholders at the top of the script:

```python
FORM_RECOGNIZER_ENDPOINT = "<Form Recognizer Endpoint URL>"
FORM_RECOGNIZER_KEY = "<Form Recognizer Key>"

AZURE_OPENAI_ENDPOINT = "<OpenAI Endpoint URL>"
AZURE_OPENAI_KEY = "<OpenAI Key>"

EMBEDDING_MODEL = "text-embedding-3-large"  # name of your embeddings deployment
CHAT_MODEL = "o4-mini"                      # name of your chat deployment
```

### Optional: use environment variables

Instead of hardcoding, you can load from environment variables:

```python
import os
FORM_RECOGNIZER_ENDPOINT = os.getenv("FR_ENDPOINT")
FORM_RECOGNIZER_KEY = os.getenv("FR_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AOAI_KEY")
```

Example `.env` (if you use `python-dotenv`):

```
FR_ENDPOINT=https://<your-form-recognizer>.cognitiveservices.azure.com/
FR_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AOAI_ENDPOINT=https://<your-azure-openai>.openai.azure.com/
AOAI_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> **Note on Azure OpenAI**: In Azure, you create **deployments** of models. The script expects the **deployment names** `text-embedding-3-large` and `o4-mini`. If your deployments use different names, set `EMBEDDING_MODEL` and `CHAT_MODEL` accordingly.

---

## Project files

```
.
├─ 2023-crv.pdf                  # source manual (you provide this)
├─ crv_manual_extracted.txt      # auto-generated: extracted text cache
├─ crv_embeddings.pkl            # auto-generated: embeddings cache
└─ crv_assistant.py              # this script (your filename)
```

Paths can be adjusted via:

```python
PDF_PATH = "2023-crv.pdf"
EXTRACTED_TEXT_PATH = "crv_manual_extracted.txt"
EMBEDDINGS_CACHE_PATH = "crv_embeddings.pkl"
```

---

## Running the assistant (CLI)

```bash
python crv_assistant.py
```

You should see:

```
🚗 Honda CR-V Assistant Ready! Type your question or 'exit' to quit.

You:
```

Try questions like:

* “How do I pair my phone with Bluetooth?”
* “What’s the recommended tire pressure?”
* “How do I use adaptive cruise control?”
* “Where is the jack located?”

Type `exit` (or `quit` / `bye`) to end.

---

## How it works (under the hood)

* **Extraction:** `DocumentAnalysisClient(...).begin_analyze_document("prebuilt-read", ...)`
* **Chunking:** simple line-accumulation up to \~500 characters (`chunk_text`)
* **Embeddings:** `client.embeddings.create(input=chunk, model=EMBEDDING_MODEL)`
* **Similarity:** `sklearn.metrics.pairwise.cosine_similarity`
* **RAG prompt:** top-K (default 3) chunks concatenated into the user message context
* **Chat:** `client.chat.completions.create(model=CHAT_MODEL, ...)` with low temperature for factual answers

---

## Customization

* **Change chunk size or strategy:** tweak `chunk_text(text, max_tokens=500)`; consider sentence boundaries or token-aware chunking for better retrieval.
* **Adjust retrieval:** tune `top_k` in `ask_crv_assistant(query, top_k=3)`.
* **Swap models:** point `EMBEDDING_MODEL` / `CHAT_MODEL` to different Azure deployments.
* **Persist chat history:** the CLI keeps a short rolling history; increase the window or store to disk if needed.

---

## Troubleshooting

* **Authentication errors**

  * Verify endpoint URLs are the service base URLs (not keys) and keys are current.
* **Model not found**

  * Ensure your **Azure OpenAI deployment names** match `EMBEDDING_MODEL` and `CHAT_MODEL`.
* **Extraction is slow or fails**

  * Very large PDFs can take time; check service region and quotas.
  * Confirm your Form Recognizer is **v3.0+** with `prebuilt-read` available.
* **Weird answers**

  * Increase `top_k` to 5; improve chunking; confirm the manual PDF is correct.
  * Lower `temperature` (already 0.2) for more deterministic replies.

---

## Security & compliance

* Keep keys out of source control (use env vars / secret managers).
* The script caches extracted text and embeddings **locally**; protect those files if they contain proprietary content.
* Respect the license of the PDF you ingest.

---

## License

Provide your project’s license here (e.g., MIT). If you are using the Honda manual, ensure personal/fair-use and do not distribute copyrighted content.

---

## Credits

* **Author:** @dsarrge
* **Title in code:** *2023 Honda CR-V Assistant*
