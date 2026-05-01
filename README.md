# local-rag-pipeline

A RAG (Retrieval-Augmented Generation) setup for querying your own documents without sending anything to an external API. Runs fully local using Ollama and LangChain.

## Why local

Didn't want my docs going to OpenAI. This uses Ollama to run the LLM on-device, so nothing leaves your machine.

## How it works

1. Drop PDFs, markdown, or text files into the `docs/` folder
2. Run the ingestion script - it chunks them, embeds with a local embedding model, and stores in ChromaDB
3. Ask questions via the CLI (`rag.py`) or the Gradio UI (`rag_app.py`)
4. Retrieves the top-k relevant chunks and feeds them into the LLM as context

## Models used

- LLM: `llama3` via Ollama (swappable for any Ollama model)
- Embeddings: `nomic-embed-text` via Ollama

## Setup

```bash
# install Ollama first: https://ollama.com
ollama pull llama3
ollama pull nomic-embed-text

pip install -r requirements.txt

# ingest your docs
python ingest.py --docs_dir ./docs

# run the CLI
python rag.py

# or run the Gradio UI
python rag_app.py
```

## Files

```
docs/            # drop your PDFs, .md, and .txt files here
ingest.py        # chunks, embeds, and indexes documents into ChromaDB
rag.py           # interactive CLI for asking questions
rag_app.py       # Gradio web UI
requirements.txt
```

## Stack

Python - LangChain - Ollama - ChromaDB - Gradio
