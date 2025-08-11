# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 18:23:47 2025
Title: 2023 Honda CR-V Assistant
Author: @dsarrge
"""
#%% === MODULES ===
"""
If package install is required, run the following commands individually in the IPython Console:
pip install azure-ai-formrecognizer
pip install azure-core
pip install scikit-learn
pip install openai
"""

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from sklearn.metrics.pairwise import cosine_similarity
from openai import AzureOpenAI
import pickle
import os

#%% === VARIABLES, KEYS & ENDPOINTS ===
FORM_RECOGNIZER_ENDPOINT = "<Form Recognizer Endpoint URL>"
FORM_RECOGNIZER_KEY = "<Form Recognizer Key>"

AZURE_OPENAI_ENDPOINT = "<OpenAI Endpoint URL>"
AZURE_OPENAI_KEY = "<OpenAI Key>"
EMBEDDING_MODEL = "text-embedding-3-large"
CHAT_MODEL = "o4-mini"

PDF_PATH = "2023-crv.pdf"
EXTRACTED_TEXT_PATH = "crv_manual_extracted.txt"
EMBEDDINGS_CACHE_PATH = "crv_embeddings.pkl"

#%% === TEXT EXTRACTION ===
if not os.path.exists(EXTRACTED_TEXT_PATH):
    print("Extracting text from PDF...")

    form_client = DocumentAnalysisClient(
        endpoint=FORM_RECOGNIZER_ENDPOINT,
        credential=AzureKeyCredential(FORM_RECOGNIZER_KEY)
        )
    
    with open(PDF_PATH, "rb") as f:
        poller = form_client.begin_analyze_document("prebuilt-read", document=f)
        result = poller.result()

    with open(EXTRACTED_TEXT_PATH, "w", encoding="utf-8") as out_file:
        for page in result.pages:
            for line in page.lines:
                out_file.write(line.content + "\n")

    print(f"Text extraction complete. Saved to: {EXTRACTED_TEXT_PATH}")
else:
    print(f"Skipping extraction. File already exists at: {EXTRACTED_TEXT_PATH}")
    
#%% === CHUNKING LOGIC ===
with open(EXTRACTED_TEXT_PATH, "r", encoding="utf-8") as f:
    full_text = f.read()

def chunk_text(text, max_tokens=500):
    chunks = []
    current = ""
    for line in text.splitlines():
        if len(current) + len(line) < max_tokens:
            current += line + "\n"
        else:
            chunks.append(current.strip())
            current = line + "\n"
    if current:
        chunks.append(current.strip())
    return chunks

chunks = chunk_text(full_text)

print(f"Total Chunks Created: {len(chunks)}")
print("\nFirst Chunk Preview:\n")
print(chunks[0][:500])

#%% === EMBEDDING CHUNKS ===
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-12-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
    )

if os.path.exists(EMBEDDINGS_CACHE_PATH):
    print(f"Loading cached embeddings from: {EMBEDDINGS_CACHE_PATH}")
    with open(EMBEDDINGS_CACHE_PATH, "rb") as f:
        embedded_chunks = pickle.load(f)
else:
    print("Generating embeddings...")
    embedded_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"Embedding chunk {i+1}/{len(chunks)}")
        resp = client.embeddings.create(
            input=chunk,
            model=EMBEDDING_MODEL
            )
        
        embedded_chunks.append((chunk, resp.data[0].embedding))
    with open(EMBEDDINGS_CACHE_PATH, "wb") as f:
        pickle.dump(embedded_chunks, f)
    print(f"Embeddings cached to: {EMBEDDINGS_CACHE_PATH}")
    
#%% === RAG PIPELINE ===
chat_history = [
    {"role": "system", "content": "You are a helpful assistant for the 2023 Honda CR-V Sport Hybrid model owner's manual."}]

def ask_crv_assistant(query, top_k=3):

    query_embed = client.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL
        ).data[0].embedding

    scores = []
    for chunk, vec in embedded_chunks:
        sim = cosine_similarity([query_embed], [vec])[0][0]
        scores.append((chunk, sim))

    top_chunks = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
    context = "\n---\n".join([c for c, _ in top_chunks])

    chat_history.append({"role": "user", "content": f"{query}\n\nContext:\n{context}"})

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=chat_history,
        temperature=0.2
        )

    reply = response.choices[0].message.content
    chat_history.append({"role": "assistant", "content": reply})
    return reply

#%% === DEPLOY AGENT (CLI) ===
print("🚗 Honda CR-V Assistant Ready! Type your question or 'exit' to quit.\n")

chat_history = []

while True:
    query = input("You: ")

    if query.lower() in {"exit", "quit", "bye"}:
        print("Assistant: 👋 Goodbye! Drive safe.")
        break

    chat_history.append({"role": "user", "content": query})
    full_context = chat_history[-5:]
    messages = [{"role": "system", "content": "You are a helpful assistant for the 2023 Honda CR-V owner's manual."}] + full_context

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        )

    reply = response.choices[0].message.content
    print(f"Assistant: {reply}\n")

    chat_history.append({"role": "assistant", "content": reply})