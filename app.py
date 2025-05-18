import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langdetect import detect
from googletrans import Translator
from gtts import gTTS
import uuid

# Set up Streamlit page config
st.set_page_config(page_title="🧘‍♂️ Spiritual Chatbot", layout="wide")

# API Key (from secrets)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# Language selection
st.sidebar.title("🌍 Language Settings")
language_map = {
    "English": "en",
    "Hindi (हिन्दी)": "hi",
    "French (Français)": "fr",
    "German (Deutsch)": "de"
}
selected_lang_name = st.sidebar.selectbox("Select your preferred response language:", list(language_map.keys()))
selected_lang = language_map[selected_lang_name]
translator = Translator()

# Prompt template
spiritual_prompt_template = PromptTemplate.from_template("""
You are a wise and thoughtful assistant. When asked about {topic}, respond with spiritual insight, drawing from ancient wisdom and the content of the document.

Use this context:
{context}

User's question:
{question}

Provide a meaningful, grounded, and topic-relevant answer.
""")

# Load PDF and prepare vector store
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
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": spiritual_prompt_template},
        return_source_documents=False
    )
    return qa_chain

# Audio generation
def generate_audio(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    filename = f"tts_{uuid.uuid4().hex}.mp3"
    tts.save(filename)
    return filename

# Session state setup
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = load_chain()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = None

# Topic selection page
if st.session_state.selected_topic is None:
    st.title("💮 How can we assist you on your spiritual journey?")
    # Wider & centered buttons
    col_spacer1, col1, col2, col3, col_spacer2 = st.columns([0.2, 1, 1, 1, 0.2])

    with col1:
        if st.button("🕊️ Moksha", use_container_width=True):
            st.session_state.selected_topic = "Moksha"
            st.experimental_rerun()
    with col2:
        if st.button("🔗 Relationships", use_container_width=True):
            st.session_state.selected_topic = "Relationships"
            st.experimental_rerun()
    with col3:
        if st.button("💼 Success", use_container_width=True):
            st.session_state.selected_topic = "Success"
            st.experimental_rerun()

# Chat page
else:
    st.title(f"🧘 {st.session_state.selected_topic}")
    if st.button("🔙 Back to Topic Selection"):
        st.session_state.selected_topic = None
        st.session_state.chat_history = []
        #st.experimental_rerun()

    for q, a in st.session_state.chat_history:
        with st.chat_message("user", avatar="👤"):
            st.markdown(q)
        with st.chat_message("assistant", avatar="🌿"):
            st.markdown(a)

    if prompt := st.chat_input(f"🌿 Ask about {st.session_state.selected_topic}..."):
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        response = st.session_state.qa_chain({
            "question": prompt,
            "chat_history": st.session_state.chat_history,
            "topic": st.session_state.selected_topic
        })
        answer = response["answer"]

        translated_answer = translator.translate(answer, src="en", dest=selected_lang).text

        with st.chat_message("assistant", avatar="🌿"):
            st.markdown(answer)
            st.markdown(f"**🔄 Translated Response ({selected_lang_name}):**")
            st.markdown(translated_answer)
            if st.button("🔈 Generate Audio", key=f"audio_btn_{len(st.session_state.chat_history)}"):
                audio_file = generate_audio(translated_answer, lang=selected_lang)
                st.audio(audio_file, format="audio/mp3")

        st.session_state.chat_history.append((prompt, translated_answer))
