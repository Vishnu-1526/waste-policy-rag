# Municipal Waste Policy Assistant 🗑️

An AI-powered chatbot that answers questions about municipal waste management policies using **RAG** (Retrieval-Augmented Generation) technology.

## 🚀 Live Demo

**[https://waste-policy-chatbot.streamlit.app](https://waste-policy-chatbot.streamlit.app)**

## 🌟 Features

- **🔍 RAG Architecture**: Retrieves relevant policy sections before generating answers
- **🧠 Llama 3.1 8B**: Powerful instruction-following LLM via HuggingFace Inference API
- **💬 Intelligent Q&A**: Ask questions about waste management policies and get accurate answers
- **✏️ Typo Tolerance**: Understands queries even with spelling mistakes
- **📚 Source References**: View the policy document sections used to generate answers
- **💾 Chat History**: Maintains conversation context during the session
- **⚡ Fast Responses**: Cached vector store for quick subsequent queries

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Meta Llama 3.1 8B Instruct** | LLM via HuggingFace Inference API (Cerebras) |
| **LangChain** | Document loading, text splitting, orchestration |
| **FAISS** | Vector database for similarity search |
| **Sentence-Transformers** | Text embeddings (`all-MiniLM-L6-v2`) |
| **Streamlit** | Web interface with chat UI |
| **PyPDF** | PDF document processing |
| **HuggingFace Hub** | Inference API client |

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
                     │  Llama 3.1   │
                     │  8B via HF   │
                     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   Final      │
                     │   Answer     │
                     └──────────────┘
```

### RAG Pipeline Steps:

1. **Document Loading**: PDF is loaded and split into 400-character chunks with 80-char overlap
2. **Embedding**: Chunks are converted to vector embeddings using Sentence-Transformers
3. **Query Processing**: User query is corrected for typos and converted to embedding
4. **Retrieval**: FAISS finds the top 3 most relevant policy chunks
5. **Prompt Building**: Retrieved context is combined with user question
6. **Generation**: Llama 3.1 8B generates a full, accurate answer via HF Inference API (Cerebras)

## ⚙️ Setup & Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Vishnu-1526/waste-policy-rag.git
cd waste-policy-rag

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your HuggingFace API key to .streamlit/secrets.toml
echo 'HF_API_KEY = "your_hf_token_here"' > .streamlit/secrets.toml

# 5. Run the app
streamlit run app.py
```

## 🔑 Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_API_KEY` | HuggingFace API token (get one free at [huggingface.co](https://huggingface.co/settings/tokens)) |

## 📁 Project Structure

```
waste-policy-rag/
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── data/
│   └── municipal_wastepolicy.pdf  # Source policy document
└── .streamlit/
    ├── config.toml            # Streamlit theme & server config
    └── secrets.toml           # API keys (gitignored)
```
