# Municipal Waste Policy Assistant 🗑️

An AI-powered chatbot that answers questions about municipal waste management policies using RAG (Retrieval-Augmented Generation) technology.

## 🌟 Features

- **Intelligent Q&A**: Ask questions about waste management policies and get accurate answers
- **Typo Tolerance**: Understands queries even with spelling mistakes
- **Source References**: View the policy document sections used to generate answers
- **Chat History**: Maintains conversation context during the session
- **Fast Responses**: Cached model loading for quick subsequent queries

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Google Flan-T5 (runs locally)
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **Framework**: LangChain
- **Document Processing**: PyPDF

The app will open at `https://waste-policy-chatbot.streamlit.app`

## 📝 How It Works

1. **Document Loading**: PDF is loaded and split into chunks
2. **Embedding**: Chunks are converted to vector embeddings
3. **Query Processing**: User query is corrected for typos and embedded
4. **Retrieval**: Most relevant chunks are retrieved using similarity search
5. **Generation**: LLM generates answer based on retrieved context
