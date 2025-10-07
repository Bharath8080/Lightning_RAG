import streamlit as st
import os, tempfile, time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Typesense
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from arize.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

tracer_provider = register(
    space_id="U3BhY2U6MjU3OTA6bXNxVA==",  
    api_key="ak-04ff1017-e4e7-47f4-893c-499d6bac22f8-mVLKZOhNZhkEMvvNTx8YkO-Kz5BdHike",
    project_name="LightiningRAG"
)
# Load .env
load_dotenv()

TS_BASE = {
    "host": os.getenv("TYPESENSE_HOST"),
    "typesense_api_key": os.getenv("TYPESENSE_API_KEY"),
    "port": "443",        # fixed
    "protocol": "https"   # fixed
}

def get_embeddings():
    return CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )

def get_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )

def process_pdfs(files, embeddings, collection_name):
    docs = []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(f.getvalue()); path = tmp.name
        docs += RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents(PyPDFLoader(path).load())
        os.unlink(path)
    return Typesense.from_documents(docs, embeddings, typesense_client_params={**TS_BASE, "typesense_collection_name": collection_name})

def build_chain(vs, llm):
    prompt = PromptTemplate(
        template="Use context to answer. If unknown, say so.\n\nContext:{context}\nHistory:{chat_history}\nQ:{question}\nA:",
        input_variables=["context", "chat_history", "question"])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm, retriever=vs.as_retriever(search_kwargs={"k": 4}),
        memory=memory, combine_docs_chain_kwargs={"prompt": prompt})

def main():
    st.set_page_config(page_title="⚡RAG Chat", page_icon="🚀")
    st.markdown(
    """
    <div style="text-align: center;">
        <h2 style="display: inline-flex; align-items: center; justify-content: center; color: white; gap: 6px;">
            ⚡Lightning RAG via 
            <span style="color: #ff6f00;">Groq</span> & 
            <span style="color: #CCFF66;">Typesense</span>
            <img src="https://avatars.githubusercontent.com/u/19822348?v=4" width="50" style="margin-left: 7px;">
        </h2>
    </div>
    """,
    unsafe_allow_html=True
)

    if "chain" not in st.session_state: st.session_state.chain = None
    if "chat_history" not in st.session_state: st.session_state.chat_history = []

    # Sidebar inputs
    st.sidebar.image("https://miro.medium.com/v2/1*b9wiAr_HG6ct7uYtCnf0xA.png", width='stretch')
    st.sidebar.markdown(
    '<hr style="border: 2px solid #ff6f00; width: 100%; margin-top: 10px; margin-bottom: 10px;">',
    unsafe_allow_html=True
)
    st.sidebar.header("⚙️ Settings")
    collection_name = st.sidebar.text_input("🔧 Collection Name", value="typesense_rag")
    files = st.sidebar.file_uploader("📄 Upload PDFs", type="pdf", accept_multiple_files=True)

    if st.sidebar.button("🚀 Process"):
        if files:
            vs = process_pdfs(files, get_embeddings(), collection_name)
            st.session_state.chain = build_chain(vs, get_llm())
            st.session_state.chat_history.clear()
            st.success(f"✅ Processed {len(files)} file(s) into collection '{collection_name}'")
        else:
            st.error("Upload at least one PDF!")

    if st.session_state.chain:
        for m in st.session_state.chat_history:
            st.chat_message(m["role"]).write(m["content"])
        q = st.chat_input("Ask about your documents...")
        if q:
            st.session_state.chat_history.append({"role": "user", "content": q})
            with st.spinner("Thinking..."):
                t0 = time.time()
                ans = st.session_state.chain({"question": q})["answer"]
                response_time = time.time() - t0
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": f"{ans}\n\n⚡ Response time: {response_time:.2f}s"
                })
            st.rerun()
    
if __name__ == "__main__":
    main()
