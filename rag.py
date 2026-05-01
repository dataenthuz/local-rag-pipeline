import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

DOCS_DIR = "./docs"
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

# Requires Ollama running locally:
#   ollama pull llama3
#   ollama pull nomic-embed-text


def build_vectorstore():
    print("Loading docs from ./docs ...")
    loader = DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    print(f"  {len(docs)} files loaded")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"  {len(chunks)} chunks created")

    print("Building embeddings (takes a minute on first run)...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_DIR
    )
    vectorstore.persist()
    return vectorstore


def load_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def build_chain(vectorstore):
    llm = Ollama(model=LLM_MODEL, temperature=0)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Use only the context below to answer the question. "
            "If the answer is not there, say so.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )


def ask(chain, question):
    result = chain({"query": question})
    sources = list({d.metadata.get("source", "?") for d in result["source_documents"]})
    print(f"\nQ: {question}")
    print(f"A: {result['result']}")
    print(f"   sources: {', '.join(sources)}")


if __name__ == "__main__":
    vs = build_vectorstore() if not os.path.exists(CHROMA_DIR) else load_vectorstore()
    chain = build_chain(vs)
    print("\nReady - Ctrl+C to quit")
    while True:
        try:
            q = input("\n> ")
            if q.strip():
                ask(chain, q)
        except KeyboardInterrupt:
            break

