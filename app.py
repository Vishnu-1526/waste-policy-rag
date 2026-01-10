import streamlit as st
import os

st.set_page_config(page_title="Municipal Waste Policy Assistant", layout="centered")

# -------------------------------
# Initialize Session State for Chat
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------
# UI Header
# -------------------------------
st.title("Municipal Waste Policy Assistant")
st.caption("Ask me anything about waste management policies!")

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
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    
    # Load documents
    loader = PyPDFLoader("data/municipal_wastepolicy.pdf")
    documents = loader.load()
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Load local LLM
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    text_generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256
    )
    
    return vectorstore, text_generator

def get_answer(query, vectorstore, text_generator):
    """Get answer using retrieval and generation"""
    # Retrieve relevant documents
    docs = vectorstore.similarity_search(query, k=4)
    
    # Combine context
    context = "\n".join([doc.page_content for doc in docs])
    
    # Create prompt
    prompt = f"""Based on the following context, answer the question accurately and concisely.
If the answer is not in the context, say "I don't have information about that."

Context:
{context}

Question: {query}

Answer:"""
    
    # Generate answer
    result = text_generator(prompt, max_new_tokens=256)
    answer = result[0]['generated_text']
    
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
        with st.spinner("Searching policies and generating answer..."):
            try:
                # Load resources (cached)
                vectorstore, text_generator = load_resources()
                
                # Correct common typos silently
                corrected_query = correct_query(prompt)
                
                # Get answer
                answer, source_docs = get_answer(corrected_query, vectorstore, text_generator)
                
                # Handle empty answers
                if not answer or len(answer.strip()) < 5:
                    answer = "I don't have specific information about that in the policy documents. Try asking about waste segregation, recycling, disposal methods, or penalties."
                
                st.markdown(answer)
                
                # Show sources (expandable)
                if source_docs:
                    with st.expander("View source references"):
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
    st.header("Suggested Questions")
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
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
