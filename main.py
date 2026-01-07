import streamlit as st
import os
import shutil
import json
import nest_asyncio
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_parse import LlamaParse


from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank

#nested asyncio loops (needed for LlamaParse)
nest_asyncio.apply()

# --- CONFIG ---
PERSIST_DIR = "./storage"
CHAT_HISTORY_FILE = "chat_history.json"
rerank_top_k = 5

st.set_page_config(page_title="Doc Reader", layout="wide")
st.title("DocuReader")

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.header("API Keys")
    groq_key = st.text_input("Groq API Key", type="password")
    llama_cloud_key = st.text_input("LlamaCloud Key (for Parsing)", type="password")
    
    if groq_key: os.environ["GROQ_API_KEY"] = groq_key
    if llama_cloud_key: os.environ["LLAMA_CLOUD_API_KEY"] = llama_cloud_key

#INITIALIZE MODELS 
if "GROQ_API_KEY" in os.environ:
    llm = Groq(model="llama-3.3-70b-versatile")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    #Re-Ranker (Runs locally on CPU)
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2", 
        top_n=rerank_top_k
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
else:
    st.warning("Please enter API keys to start.")
    st.stop()

# --- HELPER: CHAT HISTORY ---
def save_chat_history(messages):
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(messages, f)

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

# --- DOCUMENT LOADING (LlamaParse) ---
if not os.path.exists(PERSIST_DIR):
    uploaded_file = st.file_uploader("Upload a Complex PDF (Tables/Forms)", type=["pdf"])
    
    if uploaded_file and "LLAMA_CLOUD_API_KEY" in os.environ:
        with st.spinner("Analyzing document structure, Hold On"):
            # 1. Save Temp
            temp_dir = "temp_data"
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 2. PARSE WITH VISION MODEL
            # This converts tables to Markdown so the LLM understands them!
            parser = LlamaParse(result_type="markdown")
            documents = parser.load_data(file_path)
            
            # 3. Create Vector Store
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir=PERSIST_DIR)
            
            # Cleanup
            shutil.rmtree(temp_dir)
            st.rerun()


if os.path.exists(PERSIST_DIR):
    # 1. Load Vector Index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

    # 2. Setup Retrievers
    # A. Vector Retriever (Semantic)
    vector_retriever = index.as_retriever(similarity_top_k=10)
    
    # B. BM25 Retriever (Keyword)
    bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=10)

    # 3. Hybrid Fusion
    fusion_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=10, # Get top 10 candidates
        num_queries=1,       
        mode="reciprocal_rerank",
        use_async=True
    )

    # 4. The Query Engine with Re-Ranking
    query_engine = RetrieverQueryEngine.from_args(
        retriever=fusion_retriever,
        node_postprocessors=[reranker], 
        llm=llm
    )

    st.success("System Ready! Ask your questions below.")

    # --- CHAT UI ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about specific details (dates, codes, numbers)..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving & Re-ranking..."):
                response = query_engine.query(prompt)
                st.markdown(response.response)
                
                # Show Advanced Evidence
                with st.expander("Debug: Retrieval Process"):
                    for node in response.source_nodes:
                        st.caption(f"**Score:** {node.score:.4f}")
                        st.text(node.node.get_text()[:200] + "...")

                st.session_state.messages.append({"role": "assistant", "content": response.response})
                save_chat_history(st.session_state.messages)

    # --- CLEAR BUTTON ---
    if st.sidebar.button("Reset Memory"):
        if os.path.exists(PERSIST_DIR): shutil.rmtree(PERSIST_DIR)
        save_chat_history([])
        st.rerun()