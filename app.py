import streamlit as st
import os

st.set_page_config(page_title="Municipal Waste Policy Assistant", layout="centered")

# -------------------------------
# Initialize Session State for Chat
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

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
def load_qa_system():
    """Load all AI components once and cache them"""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
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
    
    # Create embeddings (semantic understanding handles typos!)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Create vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Load local LLM
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # Custom prompt for accurate, direct answers
    prompt_template = """Answer the question accurately and directly based ONLY on the provided context. 
Extract the specific information that answers the question. Be concise and precise.
If the answer is not in the context, say "I don't have information about that in the policy documents."

Context:
{context}

Question: {question}

Direct Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create QA chain with custom prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa_chain, vectorstore

# -------------------------------
# Spell Correction Helper
# -------------------------------
def correct_query(query):
    """Basic spell correction for common waste-related terms"""
    corrections = {
        # Common typos
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
        # Check if word needs correction
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
                # Load QA system (cached after first load)
                qa_chain, vectorstore = load_qa_system()
                
                # Correct common typos silently
                corrected_query = correct_query(prompt)
                
                # Get answer
                result = qa_chain.invoke({"query": corrected_query})
                answer = result["result"]
                
                # Handle empty or unhelpful answers
                if not answer or len(answer.strip()) < 10:
                    answer = "I'm not sure about that specific question. Could you try rephrasing it? You can ask me about waste segregation, recycling rules, disposal methods, penalties, or any other waste management topics!"
                
                st.markdown(answer)
                
                # Show sources (expandable)
                if "source_documents" in result and result["source_documents"]:
                    with st.expander("📄 View source references"):
                        for i, doc in enumerate(result["source_documents"][:3]):
                            st.caption(f"**Source {i+1}:** {doc.page_content[:200]}...")
                
            except Exception as e:
                answer = f"Sorry, I encountered an issue. Please try again! Error: {str(e)}"
                st.error(answer)
    
    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})

# -------------------------------
# Sidebar with examples
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