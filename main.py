import os
import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# --- Set your OpenAI API key ---
os.environ["OPENAI_API_KEY"] = "your_api_key"

# --- Streamlit UI ---
st.set_page_config(page_title="PDF Q&A", layout="wide")
st.title("💮 Ask Questions and seek answers from Guruji")

query = st.text_input("🌿 How can we assist you?")

# --- Load and process the fixed PDF ---
PDF_PATH = "data/Teachings_of_the_Bhagavadgita.pdf"

if query:
    if not os.path.exists(PDF_PATH):
        st.error(f"File not found: {PDF_PATH}")
    else:
        with st.spinner("Processing PDF..."):
            # Load PDF and split into chunks
            loader = PyPDFLoader(PDF_PATH)
            pages = loader.load()
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = splitter.split_documents(pages)

            # Create vectorstore
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(docs, embeddings)

            # Build retrieval-based QA chain
            retriever = vectorstore.as_retriever()
            llm = ChatOpenAI(model="gpt-3.5-turbo")
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

            # Get answer
            response = qa_chain.run(query)

        st.success("Answer:")
        st.markdown(f"**{response}**")

