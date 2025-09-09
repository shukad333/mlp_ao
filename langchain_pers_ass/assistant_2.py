import streamlit as st
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import wikipedia
import os

# Simple calculator tool
def calculator_tool(input_text):
    try:
        result = str(eval(input_text))
    except Exception:
        result = "Invalid expression"
    return result

# Web search tool using Wikipedia
def web_search_tool(query):
    try:
        result = wikipedia.summary(query, sentences=2)
    except Exception:
        result = "Sorry, I couldn’t find information online."
    return result

# Setup Hugging Face local model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

local_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if os.environ.get('USE_GPU') else -1,
    do_sample=True,
    max_new_tokens=150
)

llm = HuggingFacePipeline(pipeline=local_pipeline)

# Load documents and build vector store
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, 'documents', 'my_notes.txt')

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Document file not found at: {file_path}")

loader = TextLoader(file_path)
docs = loader.load()

if len(docs) == 0:
    raise ValueError("No documents loaded from my_notes.txt")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

if len(chunks) == 0:
    raise ValueError("No text chunks generated from documents")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(chunks, embeddings)

# Structured memory (last 10 interactions)
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=10
)

# Conversational retrieval chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    memory=memory
)

# Personal Assistant Agent with routing logic
class PersonalAssistantAgent:
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain

    def run(self, query):
        # Calculator route
        if any(op in query for op in ['+', '-', '*', '/']):
            return calculator_tool(query)

        # Web search route
        if query.lower().startswith(("who", "what", "when", "where", "how")):
            return web_search_tool(query)

        # Fallback to document QA chain
        return self.qa_chain.run(query)

agent = PersonalAssistantAgent(qa_chain)

# Streamlit UI
st.title("🧠 Advanced Local Personal Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask me anything:")

if st.button("Send"):
    with st.spinner("Processing... Please wait"):
        answer = agent.run(query)

    st.session_state.history.append((query, answer))

for q, a in st.session_state.history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Assistant:** {a}")
