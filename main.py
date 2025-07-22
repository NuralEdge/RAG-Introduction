# අවශ්‍ය libraries ආයාත කිරීම
import os
import streamlit as st
from PyPDF2 import PdfReader
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Groq API client ආරම්භ කිරීම
client = Groq(api_key="your-api-key")

# Streamlit UI සැකසීම
st.title("📄 PDF ප්‍රශ්න-පිළිතුරු පද්ධතිය (RAG)")
st.write("PDF ගොනුවක් උඩුගත කර ඔබේ ප්‍රශ්න අසන්න")

# PDF ගොනුව උඩුගත කිරීම
uploaded_file = st.file_uploader("PDF ගොනුව තෝරන්න", type="pdf")

if uploaded_file is not None:
    # PDF ගොනුවෙන් පාඨය උකහා ගැනීම
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  # None වලින් වළකින්න

    # පාඨය කුඩා කොටස්වලට (chunks) බෙදීම
    # CharacterTextSplitter වෙනුවට සරල බෙදීමක්
    chunk_size = 1000
    chunk_overlap = 200
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap

    # Embeddings ආරම්භ කිරීම
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)

    # FAISS vector store සැකසීම
    dimension = embeddings.shape[1]  # Embedding dimension
    index = faiss.IndexFlatL2(dimension)  # L2 දුරමිතික භාවිතය
    index.add(embeddings)  # Embeddings vector store එකට එකතු කිරීම

    # ප්‍රශ්නය ඇතුලත් කිරීම
    user_question = st.text_input("ඔබේ ප්‍රශ්නය ඇතුලත් කරන්න:")

    if user_question:
        # ප්‍රශ්නයේ embedding ගණනය කිරීම
        question_embedding = embedding_model.encode([user_question])[0]

        # FAISS හි similarity search කිරීම
        D, I = index.search(np.array([question_embedding]), k=4)  # Top 4 similar chunks
        context = "\n".join([chunks[i] for i in I[0]])

        # Groq API හරහා පිළිතුර ජනනය කිරීම
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": f"answer questions using the given source :\n\n{context}"
                },
                {
                    "role": "user",
                    "content": user_question
                }
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )

        # Streamlit හි පිළිතුර පෙන්වීම
        st.write("පිළිතුර:")
        answer_container = st.empty()
        full_answer = ""
        for chunk in completion:
            chunk_content = chunk.choices[0].delta.content or ""
            full_answer += chunk_content
            answer_container.write(full_answer)
