import streamlit as st
import os
import shutil
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


st.set_page_config(page_title="DocuReader", layout="centered")
st.title("Docu Reader")


#Sidebar 
with st.sidebar:
    st.header("Settings") 
    api_key = st.text_input("Enter your API Key", type="password") #I aint giving mine :D
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

#Setup Model
if api_key:
    os.environ["GROQ_API_KEY"] = api_key
    
    # Initialize Llama 3 via Groq Cloud
    llm = Groq(model="llama-3.3-70b-versatile") 
    
    # Small, fast embedding model (runs on CPU easily)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    Settings.llm = llm
    Settings.embed_model = embed_model
else:
    st.stop()

#Session 
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

#The RAG Pipeline
if uploaded_file and not st.session_state.query_engine:
    with st.spinner("Indexing PDF"):
        # Temp file handling
        temp_dir = "temp_data"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # Load & Indexing
        documents = SimpleDirectoryReader(temp_dir).load_data()
        index = VectorStoreIndex.from_documents(documents)
        
        # Creating Engine
        st.session_state.query_engine = index.as_chat_engine(
            chat_mode="context",
            verbose=True
        )
        st.success("Indexing Complete! Please proceed to ask whatever you want about the document from below's chatbar.")

#Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the document..."):
    #User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    #AI Response
    with st.chat_message("assistant"):
        with st.spinner("thinking..."):
            if st.session_state.query_engine:
                response = st.session_state.query_engine.chat(prompt)
                st.markdown(response.response)

                #citation scoring for showing relevace to the answer, added this because I tough it can help me understand how the model is fetching context :D
                if response.source_nodes:
                    with st.expander("View Source Context"):
                        for node in response.source_nodes:
                            st.caption(f"**Score:** {node.score:.2f}")
                            st.info(node.node.get_text()[:200] + "...")
                
                st.session_state.messages.append({"role": "assistant", "content": response.response})
            else:
                st.error("Please upload a PDF first.")