# local-rag-pipeline

A RAG (Retrieval-Augmented Generation) setup for querying your own documents without sending anything to an external API. Runs fully local using Ollama and LangChain.

## Why local

Didn't want my docs going to OpenAI. This uses Ollama to run the LLM on-device, so nothing leaves your machine.

## How it works

1. Drop PDFs or text files into the `docs/` folder
2. The ingestion script chunks them, embeds them with a local embedding model, and stores them in a ChromaDB vector store
3. Ask questions in the CLI or the simple Gradio UI
4. Retrieves the top-k relevant chunks and feeds them into the LLM as context

## Models used

- LLM: `llama3` via Ollama (swappable)
- Embeddings: `nomic-embed-text` via Ollama

## Setup

```bash
# install Ollama first: https://ollama.com
ollama pull llama3
ollama pull nomic-embed-text

pip install -r requirements.txt

# ingest your docs
python ingest.py --docs_path ./docs

# run the app
python app.py
```

## Files

```
docs/           # drop your PDFs and text files here
ingest.py       # chunks, embeds, and indexes documents
app.py          # Gradio UI for asking questions
retriever.py    # vector store query logic
requirements.txt
```

## Stack

Python · LangChain · Ollama · ChromaDB · Gradio
