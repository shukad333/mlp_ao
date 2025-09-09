import streamlit as st
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import pipeline
import os

# Setup lightweight Hugging Face model
local_pipeline = pipeline(
    "text-generation",
    model="gpt2",  # Lightweight for local usage
    device=0 if os.environ.get('USE_GPU') else -1,
    do_sample=True,
    max_length=512
)

llm = HuggingFacePipeline(pipeline=local_pipeline)

# Load documents and build vector store
loader = TextLoader('documents/my_notes.txt')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(chunks, embeddings)

# Memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Conversational QA chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    memory=memory
)

# Simple calculator tool
def calculator_tool(input_text):
    try:
        result = str(eval(input_text))
    except Exception:
        result = "Invalid expression"
    return result

# Streamlit UI
st.title("🧠 Local Personal Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask me anything:")

if st.button("Send"):
    with st.spinner("Processing... Please wait"):
        if any(op in query for op in ['+', '-', '*', '/']):
            answer = calculator_tool(query)
        else:
            answer = qa_chain.run(query)

    st.session_state.history.append((query, answer))

for q, a in st.session_state.history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Assistant:** {a}")
