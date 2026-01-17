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
st.title("🗑️ Municipal Waste Policy Assistant")
st.caption("Powered by IBM Granite + Agentic RAG | Ask me anything about waste management policies!")

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
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
    from langchain_community.vectorstores import FAISS
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools.retriever import create_retriever_tool
    from langchain.prompts import PromptTemplate
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    
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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Create retriever tool for the agent
    retriever_tool = create_retriever_tool(
        retriever,
        "waste_policy_search",
        "Search for information about municipal waste management policies. Use this tool to find rules, regulations, penalties, disposal methods, and recycling guidelines."
    )
    tools = [retriever_tool]
    
    # Load IBM Granite model
    model_id = "ibm-granite/granite-3.0-2b-instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    llm = HuggingFacePipeline(pipeline=text_generator)
    
    # Create ReAct Agent prompt
    agent_prompt = PromptTemplate.from_template("""You are a helpful Municipal Waste Policy Assistant. Answer questions about waste management policies accurately.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}""")
    
    # Create the agent
    agent = create_react_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )
    
    return vectorstore, text_generator, agent_executor, retriever

def get_answer(query, vectorstore, text_generator, agent_executor, retriever, use_agent=True):
    """Get answer using Agentic RAG or standard RAG"""
    
    if use_agent:
        try:
            # Use Agentic RAG with ReAct agent
            result = agent_executor.invoke({"input": query})
            answer = result.get("output", "")
            docs = retriever.invoke(query)
            return answer, docs
        except Exception as e:
            # Fallback to standard RAG if agent fails
            use_agent = False
    
    if not use_agent:
        # Standard RAG fallback
        docs = vectorstore.similarity_search(query, k=4)
        context = "\n".join([doc.page_content for doc in docs])
        
        prompt = f"""Based on the following context about municipal waste policies, answer the question accurately and concisely.
If the answer is not in the context, say "I don't have information about that."

Context:
{context}

Question: {query}

Answer:"""
        
        result = text_generator(prompt, max_new_tokens=256)
        answer = result[0]['generated_text']
        # Extract only the generated part after the prompt
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
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
        with st.spinner("🤖 Agent thinking and searching policies..."):
            try:
                # Load resources (cached)
                vectorstore, text_generator, agent_executor, retriever = load_resources()
                
                # Correct common typos silently
                corrected_query = correct_query(prompt)
                
                # Get answer using Agentic RAG
                answer, source_docs = get_answer(
                    corrected_query, 
                    vectorstore, 
                    text_generator, 
                    agent_executor, 
                    retriever,
                    use_agent=True
                )
                
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
    st.header("🤖 About This App")
    st.markdown("""
    **Technologies Used:**
    - 🧠 **IBM Granite 3.0** - Advanced LLM
    - 🔄 **Agentic RAG** - Multi-step reasoning
    - 🔍 **FAISS** - Vector search
    - 📊 **LangChain** - AI orchestration
    """)
    
    st.divider()
    
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
