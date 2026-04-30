import argparse
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "./chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_documents(docs_dir):
    docs = []
    for f in Path(docs_dir).rglob("*"):
        if f.suffix == ".pdf":
            loader = PyPDFLoader(str(f))
        elif f.suffix == ".md":
            loader = UnstructuredMarkdownLoader(str(f))
        elif f.suffix == ".txt":
            loader = TextLoader(str(f), encoding="utf-8")
        else:
            continue
        loaded = loader.load()
        docs.extend(loaded)
        print(f"Loaded: {f.name} ({len(loaded)} pages)")
    return docs


def main(docs_dir):
    print(f"Loading from: {docs_dir}")
    documents = load_documents(docs_dir)
    if not documents:
        print("No documents found.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    print("Generating embeddings via Ollama (nomic-embed-text)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH
    )
    vectorstore.persist()
    print(f"Done. {vectorstore._collection.count()} vectors stored in {CHROMA_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", default="./docs")
    args = parser.parse_args()
    main(args.docs_dir)
