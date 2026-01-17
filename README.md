# Municipal Waste Policy Assistant 🗑️

An AI-powered chatbot that answers questions about municipal waste management policies using **RAG** (Retrieval-Augmented Generation) technology.

## 🚀 Live Demo

**[https://waste-policy-chatbot.streamlit.app](https://waste-policy-chatbot.streamlit.app)**

## 🌟 Features

- **🔍 RAG Architecture**: Retrieves relevant policy sections before generating answers
- **🧠 Google FLAN-T5**: Lightweight language model for accurate responses
- **💬 Intelligent Q&A**: Ask questions about waste management policies and get accurate answers
- **✏️ Typo Tolerance**: Understands queries even with spelling mistakes
- **📚 Source References**: View the policy document sections used to generate answers
- **💾 Chat History**: Maintains conversation context during the session
- **⚡ Fast Responses**: Cached model loading for quick subsequent queries

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Google FLAN-T5** | Large Language Model for text generation |
| **LangChain** | Document loading, text splitting, orchestration |
| **FAISS** | Vector database for similarity search |
| **Sentence-Transformers** | Text embeddings (all-MiniLM-L6-v2) |
| **Streamlit** | Web interface with chat UI |
| **PyPDF** | PDF document processing |

## 🔄 How RAG Works

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   User       │ ──▶ │   Spell      │ ──▶ │  Embedding   │
│   Question   │     │   Correction │     │   Model      │
└──────────────┘     └──────────────┘     └──────────────┘
                                                │
                                                ▼
                     ┌──────────────┐     ┌──────────────┐
                     │   Prompt     │ ◀── │   FAISS      │
                     │   Builder    │     │   Search     │
                     └──────────────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   FLAN-T5    │
                     │   Generation │
                     └──────────────┘
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
3. **Query Processing**: User query is corrected for typos and converted to embedding
4. **Retrieval**: FAISS finds the top 4 most relevant policy chunks
5. **Prompt Building**: Retrieved context is combined with user question
6. **Generation**: FLAN-T5 generates answer based on retrieved context
