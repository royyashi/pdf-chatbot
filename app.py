import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }

    .stTextInput > div > div > input {
        background-color: #262730;
        color: white;
        border-radius: 10px;
        padding: 10px;
    }

    .stFileUploader {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
    }

    .chat-box {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        border: 1px solid #333;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #B0B0B0;
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# OpenRouter client
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Header
st.markdown(
    '<div class="title">📄 AI PDF Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload your PDF and ask intelligent questions instantly</div>',
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    st.markdown("---")

    st.write("### 🤖 Model")
    st.write("Using OpenRouter Free Model")

    st.markdown("---")

    st.write("### 📌 Features")
    st.write("✅ PDF Upload")
    st.write("✅ AI Question Answering")
    st.write("✅ Semantic Search")
    st.write("✅ Vector Embeddings")

    st.markdown("---")

    st.info("Built with Streamlit + OpenRouter + FAISS")

# Upload PDF
uploaded_file = st.file_uploader(
    "📂 Upload your PDF",
    type="pdf"
)

if uploaded_file is not None:

    with st.spinner("📖 Reading PDF..."):

        # Read PDF
        pdf_reader = PdfReader(uploaded_file)

        text = ""

        for page in pdf_reader.pages:
            extracted_text = page.extract_text()

            if extracted_text:
                text += extracted_text

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = text_splitter.split_text(text)

        # Create embeddings
        embeddings = embedding_model.encode(chunks)

        # Store embeddings in FAISS
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)

        index.add(np.array(embeddings))

    st.success("✅ PDF processed successfully!")

    # Question input
    question = st.text_input(
        "💬 Ask a question from the PDF"
    )

    if question:

        with st.spinner("🤖 Thinking..."):

            # Embed question
            question_embedding = embedding_model.encode([question])

            # Search similar chunks
            k = 3

            distances, indices = index.search(
                np.array(question_embedding),
                k
            )

            # Relevant chunks
            relevant_chunks = []

            for idx in indices[0]:
                relevant_chunks.append(chunks[idx])

            context = "\n".join(relevant_chunks)

            # Prompt
            prompt = f"""
            Answer the question using the context below.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """

            try:

                # Generate response
                response = client.chat.completions.create(
                    model="openrouter/free",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                # Extract answer
                answer = response.choices[0].message.content

                # Styled output box

                st.markdown("### 🙋 Your Question")
                st.info(question)

                st.markdown("### 🤖 AI Answer")
                st.success(answer)

                

                # Download answer button
                st.download_button(
                    "📥 Download Answer",
                    answer,
                    file_name="answer.txt"
                )

            except Exception as e:
                st.error(f"Error: {e}")