# Municipal Waste Policy Assistant 🗑️

An AI-powered chatbot that answers questions about municipal waste management policies using **Agentic RAG** (Retrieval-Augmented Generation) technology powered by **IBM Granite**.

## 🚀 Live Demo

**[https://waste-policy-chatbot.streamlit.app](https://waste-policy-chatbot.streamlit.app)**

## 🌟 Features

- **🤖 Agentic RAG**: Multi-step reasoning with LangChain ReAct agents
- **🧠 IBM Granite 3.0**: State-of-the-art language model for accurate responses
- **💬 Intelligent Q&A**: Ask questions about waste management policies and get accurate answers
- **✏️ Typo Tolerance**: Understands queries even with spelling mistakes
- **📚 Source References**: View the policy document sections used to generate answers
- **💾 Chat History**: Maintains conversation context during the session
- **⚡ Fast Responses**: Cached model loading for quick subsequent queries

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **IBM Granite 3.0** | Large Language Model for generation |
| **LangChain Agents** | Agentic RAG with ReAct reasoning |
| **FAISS** | Vector database for similarity search |
| **Sentence-Transformers** | Text embeddings (all-MiniLM-L6-v2) |
| **Streamlit** | Web interface |
| **PyPDF** | PDF document processing |

## 🔄 How Agentic RAG Works

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   User       │ ──▶ │   ReAct      │ ──▶ │  Retriever   │
│   Question   │     │   Agent      │     │    Tool      │
└──────────────┘     └──────────────┘     └──────────────┘
                            │                     │
                            ▼                     ▼
                     ┌──────────────┐     ┌──────────────┐
                     │   Thought    │     │   FAISS      │
                     │   Process    │     │   Search     │
                     └──────────────┘     └──────────────┘
                            │                     │
                            ▼                     ▼
                     ┌──────────────┐     ┌──────────────┐
                     │   IBM        │ ◀── │  Retrieved   │
                     │   Granite    │     │   Context    │
                     └──────────────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   Final      │
                     │   Answer     │
                     └──────────────┘
```

### RAG Pipeline Steps:

1. **Document Loading**: PDF is loaded and split into 500-character chunks
2. **Embedding**: Chunks are converted to vector embeddings using Sentence-Transformers
3. **Agent Initialization**: ReAct agent is created with retriever tool
4. **Query Processing**: User query is corrected for typos
5. **Agentic Reasoning**: Agent thinks, acts, and observes in a loop
6. **Retrieval**: Agent uses retriever tool to find relevant policy chunks
7. **Generation**: IBM Granite generates answer based on retrieved context
