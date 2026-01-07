#  Basic RAG Document reader
* **Update:** 
**Added more features for more efficeint RAG**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-Data_Framework-orange?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-LPU_Inference-purple?style=for-the-badge)

> 

---

##  Why did I make this Project?

Just to learn how RAG works by making a RAG pipeline using Groq 

**I tried to implement the following standards:**
* **Hyper-Speed Inference:** Utilizing **Groq's LPU (Language Processing Unit)** hardware to run llama-3.3-70b at 300+ tokens/second.
* **Evidence-Based Answers:** Implementing a **Citation Layer**. Every answer includes a "Check Sources" dropdown that maps the response back to the exact text chunk in the PDF.
* **Privacy-Centric Architecture:** Embedding (vectorization) happens locally using `HuggingFace`, ensuring your raw data vectors aren't unnecessarily shipped to closed-source embedding providers.
* **Simple yet efficient Steamlit UI:** Utilizing python streamlit's virsitility and efficiency to design & deploy fast 

---
##  Why This Upgrade?

Most basic RAG tutorials fail in the real world because they cannot read tables and miss exact keywords (like part numbers or dates). This project implements the **"Advanced RAG Stack"**, to fix that:

* **Complex Document Parsing:** Replaced standard PDF readers with **LlamaParse**, a vision-based model that converts PDFs into clean Markdown, preserving tables and headers perfectly.
* **Hybrid Search (The "Best of Both" Retrieval):** Combines **BM25 (Keyword Search)** with **Vector Search (Semantic Search)** using Reciprocal Rank Fusion. This ensures queries like *"Error code 404"* (Exact) and *"System failure"* (Semantic) both work.
* **Cross-Encoder Re-Ranking:** Implemented a verification step where a high-accuracy model (`ms-marco-MiniLM`) re-reads the retrieved chunks and discards irrelevant ones before they reach the LLM.
* **State Persistence:** Includes local storage for the Vector Index and Chat History, allowing users to restart the app without losing data or re-processing files.

---

##  Architecture

The system follows a standard RAG workflow but optimized for speed:

```mermaid
graph TD
    subgraph Ingestion
    A[User PDF] -->|LlamaParse| B("Structured Markdown")
    B -->|Embedding| C["Vector Store & Doc Store"]
    end

    subgraph Retrieval
    Q[User Query] -->|Semantic Search| D["Vector Retriever"]
    Q -->|Keyword Search| E["BM25 Retriever"]
    D & E -->|Reciprocal Rank Fusion| F["Candidate Chunks"]
    end

    subgraph Refinement
    F -->|Cross-Encoder Model| G{"Re-Ranker"}
    G -->|Filter Top-5| H["High-Quality Context"]
    end

    subgraph Generation
    H -->|Context + Query| I["Groq Cloud (Llama 3)"]
    I -->|Answer + Citations| J["User UI"]
    end