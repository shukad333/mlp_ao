from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# 1️⃣ Load PDF
loader = PyPDFLoader('sample.pdf')
docs = loader.load()

# 2️⃣ Split Text
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# 3️⃣ Build Vector Store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(chunks, embeddings)

# 4️⃣ Build QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

# 5️⃣ Ask Question
query = "What is the summary of this document?"
answer = qa_chain.run(query)
print(answer)
