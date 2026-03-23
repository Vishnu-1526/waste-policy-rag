import streamlit as st
import os
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Municipal Waste Policy Assistant", layout="centered")

# -------------------------------
# Initialize Session State for Chat
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# UI Header
# -------------------------------
st.title("🗑️ Municipal Waste Policy Assistant")
st.caption("Powered by RAG | Ask me anything about waste management policies!")

# -------------------------------
# Hugging Face Token
# -------------------------------
if "HF_API_KEY" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HF_API_KEY"]
elif os.getenv("HF_API_KEY"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_API_KEY")

# -------------------------------
# Load Resources Once (Cached)
# -------------------------------
@st.cache_resource
def load_resources():
    """Load all AI components once and cache them"""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    # Load documents
    loader = PyPDFLoader("data/municipal_wastepolicy.pdf")
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80
    )
    chunks = splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


def query_llm(prompt, hf_token):
    """Call zephyr-7b-beta via HF InferenceClient (chat_completion) — confirmed working."""
    client = InferenceClient(token=hf_token)
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions about municipal waste management policies. Always answer based on the provided context."},
            {"role": "user", "content": prompt}
        ],
        model="HuggingFaceH4/zephyr-7b-beta",
        max_tokens=300,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def get_answer(query, vectorstore):
    """Get answer using RAG"""
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")

    # Retrieve relevant documents
    docs = vectorstore.similarity_search(query, k=3)

    # Combine context
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""Use the following context from the municipal waste policy document to answer the question.

Context:
{context}

Question: {query}"""

    answer = query_llm(prompt, hf_token)
    return answer, docs

# -------------------------------
# Spell Correction Helper
# -------------------------------
def correct_query(query):
    """Basic spell correction for common waste-related terms"""
    corrections = {
        "wast": "waste", "wsate": "waste",
        "recycel": "recycle", "recyle": "recycle",
        "segreation": "segregation", "segregaton": "segregation", "segragation": "segregation",
        "composte": "compost", "compst": "compost",
        "garbag": "garbage", "grabage": "garbage",
        "disposl": "disposal", "disposla": "disposal",
        "hazardus": "hazardous", "hazardos": "hazardous", "hazrdous": "hazardous",
        "plastik": "plastic", "plasitc": "plastic",
        "penality": "penalty", "penalti": "penalty",
        "municpal": "municipal", "municipl": "municipal",
        "polluton": "pollution", "polution": "pollution",
        "enviornment": "environment", "enviroment": "environment",
        "collecton": "collection", "collectin": "collection",
        "managment": "management", "managemnt": "management",
    }
    
    words = query.lower().split()
    corrected_words = []
    
    for word in words:
        corrected = word
        for typo, correct in corrections.items():
            if typo in word:
                corrected = word.replace(typo, correct)
                break
        corrected_words.append(corrected)
    
    return " ".join(corrected_words)

# -------------------------------
# Display Chat History
# -------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------------
# Chat Input
# -------------------------------
if prompt := st.chat_input("Ask a question about waste policies..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching policies and generating answer..."):
            try:
                # Load resources (cached)
                vectorstore = load_resources()

                # Correct common typos silently
                corrected_query = correct_query(prompt)

                # Get answer
                answer, source_docs = get_answer(corrected_query, vectorstore)
                
                # Handle empty answers
                if not answer or len(answer.strip()) < 5:
                    answer = "I don't have specific information about that in the policy documents. Try asking about waste segregation, recycling, disposal methods, or penalties."
                
                st.markdown(answer)
                
                # Show sources (expandable)
                if source_docs:
                    with st.expander("📚 View source references"):
                        for i, doc in enumerate(source_docs[:3]):
                            st.caption(f"**Source {i+1}:** {doc.page_content[:200]}...")
                
            except Exception as e:
                answer = f"Sorry, I encountered an issue. Please try again! Error: {str(e)}"
                st.error(answer)
    
    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})

# -------------------------------
# Sidebar with suggestions
# -------------------------------
with st.sidebar:
    st.header("💡 Suggested Questions")
    st.markdown("""
    **Waste Segregation:**
    - What is waste segregation at source?
    - How to separate wet and dry waste?
    
    **Disposal & Collection:**
    - How to dispose hazardous waste?
    - What are the collection timings?
    
    **Rules & Penalties:**
    - What are the penalties for littering?
    - What are the rules for bulk waste?
    
    **Recycling:**
    - How to recycle plastic waste?
    - What items can be recycled?
    """)
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
