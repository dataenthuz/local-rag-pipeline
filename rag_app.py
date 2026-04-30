"""
Local RAG app - Ollama + LangChain + ChromaDB + Gradio.
Runs fully local, no API keys needed.

Setup:
    ollama pull llama3
    ollama pull nomic-embed-text
    python ingest.py --docs_dir ./docs
    python rag_app.py
"""

import gradio as gr
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

CHROMA_PATH = "./chroma_db"
LLM_MODEL = "llama3"
EMBED_MODEL = "nomic-embed-text"
TOP_K = 4

PROMPT_TEMPLATE = """Answer the question using only the context provided.
If the answer is not in the context, say so - do not make things up.

Context:
{context}

Question: {question}

Answer:"""


def load_chain():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    llm = Ollama(model=LLM_MODEL, temperature=0.1)
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )


def answer(question, history):
    if not question.strip():
        return history, ""
    try:
        result = chain({"query": question})
        ans = result["result"].strip()
        sources = list({doc.metadata.get("source", "unknown") for doc in result["source_documents"]})
        if sources:
            ans += f"\n\n*Sources: {', '.join(sources)}*"
        history.append((question, ans))
        return history, ""
    except Exception as e:
        history.append((question, f"Error: {e}"))
        return history, ""


print("Loading chain...")
chain = load_chain()
print("Ready at http://localhost:7860")

with gr.Blocks(title="Local RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## Local RAG - Ask questions about your documents")
    chatbot = gr.Chatbot(height=420)
    with gr.Row():
        q = gr.Textbox(placeholder="Ask something...", show_label=False, scale=4)
        gr.Button("Ask", variant="primary", scale=1).click(answer, [q, chatbot], [chatbot, q])
    q.submit(answer, [q, chatbot], [chatbot, q])
    gr.Button("Clear", size="sm").click(lambda: ([], ""), outputs=[chatbot, q])

demo.launch(server_port=7860)
