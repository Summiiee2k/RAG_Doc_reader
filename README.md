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

##  Architecture

The system follows a standard RAG workflow but optimized for speed:

```mermaid
graph LR
    A[User PDF] -->|Ingest| B("Streamlit UI")
    B -->|Local Embedding| C["HuggingFace BGE-Small"]
    C -->|Vectorize| D[("Vector Store")]
    E[User Query] -->|Search| D
    D -->|Retrieve Top-k Chunks| F["Context Window"]
    F -->|Send to Cloud| G["Groq Cloud (Llama 3)"]
    G -->|Response + Citations| H["User UI"]