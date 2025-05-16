# 🧘 Spiritual Teachings Chatbot - PDF + RAG Version with Local LLM + Gradio UI (CPU Friendly)

# Requirements:
# pip install sentence-transformers faiss-cpu pymupdf transformers gradio

import faiss
import fitz  # PyMuPDF
import numpy as np
import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import gradio as gr

# ----------------------
# Load Embedding Model
# ----------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ----------------------
# Load and Extract Text from PDF
# ----------------------
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text_blocks = []
    for page in doc:
        text_blocks.append(page.get_text())
    doc.close()
    return "\n".join(text_blocks)

# ----------------------
# Chunk Text into Paragraphs
# ----------------------
def chunk_text(text, max_length=500):
    chunks = text.split("\n\n")
    return [chunk.strip() for chunk in chunks if 30 < len(chunk.strip()) < max_length]

# ----------------------
# Load, Embed, and Index
# ----------------------
def load_pdf_and_build_index(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings))
    return chunks, index, text

# ----------------------
# Load Local LLM (CPU-Friendly Model)
# ----------------------
def load_local_llm():
    model_name = "sshleifer/tiny-gpt2"  # Extremely small and CPU-friendly
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    return pipe

llm_pipe = load_local_llm()

# ----------------------
# Query LLM with Context
# ----------------------
def query_llm_with_context(query, context):
    prompt = f"""
You are a wise assistant answering questions based only on the teachings of saints.
Use the context provided below to answer the user's question.

Context:
{context}

Question: {query}
Answer:
"""
    response = llm_pipe(prompt, max_new_tokens=150, do_sample=True)[0]['generated_text']
    return response.split("Answer:")[-1].strip()

# ----------------------
# Gradio UI
# ----------------------
pdf_path = "data/Bhagavad-gita.pdf"
chunks, index, full_text = [], None, ""

if os.path.exists(pdf_path):
    chunks, index, full_text = load_pdf_and_build_index(pdf_path)

    def answer_question(user_query):
        query_vec = model.encode([user_query])
        D, I = index.search(np.array(query_vec), k=3)
        top_chunks = [chunks[i] for i in I[0]]
        context = "\n\n".join(top_chunks)
        return query_llm_with_context(user_query, context)

    interface = gr.Interface(
        fn=answer_question,
        inputs=gr.Textbox(lines=2, placeholder="Ask a spiritual question..."),
        outputs=gr.Textbox(label="Answer"),
        title="🧘 Saints' Teachings Chatbot (Tiny GPT-2)",
        description="Ask a question, and the assistant will respond using only the teachings from your uploaded PDF."
    )

    interface.launch(debug=True)
else:
    print("PDF not found in 'data/' folder. Please add 'saint_teachings.pdf'.")