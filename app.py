import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langdetect import detect
from googletrans import Translator
from gtts import gTTS
import uuid

# Set API Key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# UI
st.set_page_config(page_title="PDF Q&A", layout="wide")
st.title("💮 Ask Questions and seek answers from Guruji")

# Language selector in sidebar
st.sidebar.title("🌍 Language Settings")
language_map = {
    "English": "en",
    "Hindi (हिन्दी)": "hi",
    "French (Français)": "fr",
    "German (Deutsch)": "de"
}
selected_lang_name = st.sidebar.selectbox("Select your preferred response language:", list(language_map.keys()))
selected_lang = language_map[selected_lang_name]

# Translator
translator = Translator()

# Load and embed PDF once
@st.cache_resource
def load_chain(pdf_path="data/Teachings_of_the_Bhagavadgita.pdf"):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(docs, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, return_source_documents=False)
    return qa_chain

def generate_audio(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    filename = f"tts_{uuid.uuid4().hex}.mp3"
    tts.save(filename)
    return filename

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = load_chain()

# Display chat history
for i, (question, answer) in enumerate(st.session_state.chat_history):
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="🌿"):
        st.markdown(answer)

# User input
if prompt := st.chat_input("🌿 How can we assist you?"):
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Get answer from LangChain
    with st.chat_message("assistant", avatar="🌿"):
        response = st.session_state.qa_chain({"question": prompt, "chat_history": st.session_state.chat_history})
        answer = response["answer"]
        st.markdown(answer)

    # Translate back to selected language
    translated_answer = translator.translate(answer, src="en", dest=selected_lang).text

    lang_code = selected_lang  # 'en', 'hi', etc.
    audio_file = generate_audio(translated_answer, lang=lang_code)
    st.audio(audio_file, format="audio/mp3")

    with st.chat_message("assistant"):
        st.markdown(translated_answer)

    # Store in session history
    st.session_state.chat_history.append((prompt, answer))
