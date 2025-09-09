import streamlit as st
import tempfile
import os
import openai
import shutil
import faiss
import pandas as pd
import pytesseract
from PIL import Image
import hashlib
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
import easyocr

# ---------------------- Load Environment Variables ----------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------------------- Configure Tesseract ----------------------
pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or r"C:\\Users\\akansha.khandare\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"

# EasyOCR fallback
reader = easyocr.Reader(["en"], gpu=False)

# ---------------------- Streamlit Config ----------------------
st.set_page_config(page_title="Multimodal Bot", page_icon="🎓", layout="wide")
st.title("🎓 Multimodal Academic Bot")

# ---------------------- Helpers ----------------------
def get_text_hash(text: str) -> str:
    """Return a hash of text for caching."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def extract_text_from_file(file):
    """Extract text or visual understanding from uploaded file depending on type."""
    import io

    ext = os.path.splitext(file.name)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    try:
        # PDF
        if ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            return "\n".join([d.page_content for d in docs])

        # Word document
        elif ext == ".docx":
            from docx import Document
            doc = Document(tmp_path)
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

        # Text file
        elif ext == ".txt":
            with open(tmp_path, "r", encoding="utf-8") as f:
                return f.read()

        # CSV
        elif ext == ".csv":
            df = pd.read_csv(tmp_path)
            return df.head(20).to_string()

        # Excel
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(tmp_path)
            return df.head(20).to_string()

        # Images: Use GPT-4o Vision first, fallback to OCR
        elif ext in [".jpg", ".jpeg", ".png"]:
            try:
                with open(tmp_path, "rb") as f:
                    response = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are an assistant that extracts and explains text, tables, charts, and visual data from images."},
                            {"role": "user", "content": "Analyze this image and summarize all important text, tables, charts, and insights."}
                        ],
                        input=[{"type": "input_image", "image": f}]
                    )
                return response.choices[0].message.content
            except Exception as e:
                st.warning(f"⚠️ GPT-4o Vision failed, using OCR fallback: {e}")
                try:
                    return pytesseract.image_to_string(Image.open(tmp_path))
                except:
                    return " ".join(reader.readtext(tmp_path, detail=0))
        else:
            st.warning(f"⚠️ Unsupported file type: {ext}")
            return ""

    except Exception as e:
        st.error(f"❌ Error reading file {file.name}: {e}")
        return ""

# ---------------------- Caching ----------------------
@st.cache_resource
def build_vectorstore(all_text, text_hash):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.create_documents([all_text])
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(split_docs, embeddings)

@st.cache_resource
def get_qa_chain(text_hash, _vectorstore):
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})
    chat_model = ChatOpenAI(model="gpt-4o", temperature=0)
    return ConversationalRetrievalChain.from_llm(llm=chat_model, retriever=retriever)

@st.cache_data
def generate_tts_audio(text, filename="answer.mp3"):
    speech_file_path = Path(filename)
    with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text
    ) as response:
        response.stream_to_file(speech_file_path)
    return str(speech_file_path)

# ---------------------- Session State ----------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_files" not in st.session_state:
    st.session_state.last_files = []
if "latest_answer" not in st.session_state:
    st.session_state.latest_answer = ""
if "speak_answer" not in st.session_state:
    st.session_state.speak_answer = False

# ---------------------- File Upload ----------------------
uploaded_files = st.file_uploader(
    "📂 Upload guidelines, reports, images, spreadsheets (multiple files)",
    type=["pdf", "docx", "txt", "csv", "xls", "xlsx", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# ---------------------- Reset if New Files Uploaded ----------------------
current_files = [file.name for file in uploaded_files] if uploaded_files else []
if current_files != st.session_state.last_files:
    st.session_state.chat_history = []
    st.session_state.last_files = current_files
    st.session_state.latest_answer = ""
    st.session_state.speak_answer = False

# ---------------------- Process Uploaded Files ----------------------
if uploaded_files:
    all_text = ""
    for file in uploaded_files:
        content = extract_text_from_file(file)
        if content:
            all_text += content + "\n"

    if all_text.strip():
        text_hash = get_text_hash(all_text)

        with st.spinner("🔍 Processing and indexing files..."):
            vectorstore = build_vectorstore(all_text, text_hash)
            qa_chain = get_qa_chain(text_hash, vectorstore)

        st.success("✅ Files processed and indexed!")

        # ---------------------- Input Method ----------------------
        st.subheader("Ask a Question")
        input_method = st.radio("Choose input method:", ["Type", "Speak"])

        query = ""

        # Typing Mode
        if input_method == "Type":
            query = st.text_input(
                "💬 Type your question and press Enter",
                key=f"user_query_{len(st.session_state.chat_history)}",
                placeholder="Ask something about your uploaded documents..."
            )

        # Speaking Mode
        elif input_method == "Speak":
            st.write("🎤 Record your question:")
            audio_data = mic_recorder(
                start_prompt="🎙️ Start Recording",
                stop_prompt="⏹️ Stop",
                key=f"recorder_{len(st.session_state.chat_history)}"
            )

            if audio_data and "bytes" in audio_data:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_data["bytes"])
                    tmp_path = tmp.name

                with open(tmp_path, "rb") as f:
                    transcript = openai.audio.transcriptions.create(
                        model="whisper-1",
                        file=f
                    )
                query = transcript.text
                st.success(f"✅ Recognized Speech: {query}")

        # ---------------------- Run Query ----------------------
        if query.strip():
            with st.spinner("🤖 Generating answer..."):
                result = qa_chain.invoke({
                    "question": query,
                    "chat_history": st.session_state.chat_history
                })

            answer = result["answer"] if isinstance(result, dict) else result

            # Save chat
            st.session_state.chat_history.append((query, answer))
            st.session_state.latest_answer = answer
            st.session_state.speak_answer = False

        # ---------------------- Show Conversation ----------------------
        if st.session_state.chat_history:
            st.markdown("### 📝 Conversation")
            for i, (q, a) in enumerate(st.session_state.chat_history, start=1):
                st.markdown(f"""
                <div style="margin-bottom: 15px; padding: 10px; border-radius: 10px; background-color: #e3f2fd;">
                    <b style="color:#1565c0;">Q{i}:</b> {q}
                </div>
                <div style="margin-bottom: 15px; padding: 10px; border-radius: 10px; background-color: #f5f5f5;">
                    <b style="color:#2e7d32;">A{i}:</b> {a}
                </div>
                """, unsafe_allow_html=True)

        # ---------------------- Speak Latest Answer ----------------------
        if st.session_state.latest_answer:
            speak_check = st.checkbox(
                "🔊 Speak Latest Answer",
                value=st.session_state.speak_answer,
                key=f"speak_{len(st.session_state.chat_history)}"
            )

            if speak_check:
                audio_file = generate_tts_audio(st.session_state.latest_answer)
                st.audio(audio_file, format="audio/mp3")
                st.session_state.speak_answer = True
            else:
                st.session_state.speak_answer = False
