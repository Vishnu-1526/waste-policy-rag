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

## 📋 Prerequisites

- Python 3.9+
- pip package manager

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Vishnu-1526/waste-policy-rag.git
   cd waste-policy-rag
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your PDF document**
   - Place your waste policy PDF in the `data/` folder
   - Name it `municipal_wastepolicy.pdf`

## 💻 Running Locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 🌐 Deployment

### Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the main file path to `app.py`
5. Add secrets in Settings → Secrets:
   ```toml
   HF_API_KEY = "your_huggingface_token"
   ```
6. Deploy!

## 📁 Project Structure

```
waste-policy-rag/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── data/
│   └── municipal_wastepolicy.pdf  # Policy document
├── .gitignore
└── README.md
```

## 💡 Example Questions

- What is waste segregation at source?
- How to dispose hazardous waste?
- What are the penalties for littering?
- How to recycle plastic waste?
- What are the collection timings?

## 🔧 Configuration

The app automatically handles:
- Document chunking (500 characters with 100 overlap)
- Semantic search with top 4 relevant chunks
- Response generation with max 256 tokens

## 📝 How It Works

1. **Document Loading**: PDF is loaded and split into chunks
2. **Embedding**: Chunks are converted to vector embeddings
3. **Query Processing**: User query is corrected for typos and embedded
4. **Retrieval**: Most relevant chunks are retrieved using similarity search
5. **Generation**: LLM generates answer based on retrieved context

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Vishnu**
- GitHub: [@Vishnu-1526](https://github.com/Vishnu-1526)

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [LangChain](https://langchain.com/) for RAG components
- [Hugging Face](https://huggingface.co/) for transformers and models
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search

---

⭐ Star this repo if you find it helpful!
