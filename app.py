import streamlit as st
import os
import re
import tempfile
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import time
from pathlib import Path
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("⚠ Google API Key is missing! Please set it in your environment variables.")
    st.stop()
else:
    genai.configure(api_key=API_KEY)

# Streamlit Page Configuration
st.set_page_config(page_title="Multimodal AI Chat", layout="wide")
st.title("📚 Chat with YouTube, PDF or Local Video")

# Initialize Agent for Local Video Analysis
@st.cache_resource
def initialize_agent():
    return Agent(
        name="Video AI Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

multimodal_Agent = initialize_agent()

# Function to extract video ID from YouTube URLs
def extract_video_id(url):
    patterns = [
        r"v=([a-zA-Z0-9_-]+)", 
        r"youtu\.be/([a-zA-Z0-9_-]+)",  
        r"youtube\.com/shorts/([a-zA-Z0-9_-]+)", 
        r"youtube\.com/embed/([a-zA-Z0-9_-]+)"  
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# Function to get YouTube transcript
def extract_transcript_details(youtube_video_url):
    try:
        video_id = extract_video_id(youtube_video_url)
        if not video_id:
            return "Invalid YouTube URL."
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry["text"] for entry in transcript_data])
        return transcript_text
    except TranscriptsDisabled:
        return "Transcript is not available for this video."
    except Exception as e:
        return f"Error retrieving transcript: {str(e)}"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(pdf_file.read())
            temp_file_path = temp_file.name
        text = ""
        with open(temp_file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        os.remove(temp_file_path)
        return text.strip()
    except Exception as e:
        return f"Error extracting text from PDF: {e}"

# Function to process text for Q&A
def process_text_for_chat(text, storage_name):
    text_chunks = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200).split_text(text)
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(storage_name)
        return "✅ Data processed! Now ask questions."
    except Exception as e:
        return f"❌ Error processing text: {e}"

# Function to get answer from processed data
def get_answer(query, storage_name):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local(storage_name, embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(query)

        if not docs:
            return f"🤔 I couldn't find an answer. Try rephrasing your question."
        
       

        prompt_template = """
        Answer the question in detail based on the provided document content. If the answer is not found, say:
        "I couldn't find the answer in the document."

        Document Context:\n {context}\n
        Question: \n{question}\n
        Answer:
        """

        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

        response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
        return response["output_text"]

    except Exception as e:
        return f"⚠ Error processing query: {e}"

# Streamlit UI for mode selection
mode = st.radio("Choose an option:", ["📄 Chat with PDF", "🎥 Chat with YouTube Video", "🎬 Chat with Local Video"])

# PDF chat
if mode == "📄 Chat with PDF":
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if pdf_file:
        with st.spinner("⏳ Extracting text..."):
            pdf_text = extract_text_from_pdf(pdf_file)
        if "Error" not in pdf_text:
            with st.spinner("⏳ Processing text for Q&A..."):
                process_status = process_text_for_chat(pdf_text, "faiss_pdf")
                st.success(process_status)

    user_question = st.text_input("Ask a question about the PDF:")
    if user_question:
        response = get_answer(user_question, "faiss_pdf")
        st.markdown(f"🧑‍💻 You:** {user_question}")
        st.markdown(f"🤖 AI:** {response}")

# YouTube chat
elif mode == "🎥 Chat with YouTube Video":
    youtube_link = st.text_input("Enter YouTube Video Link:")
    if youtube_link:
        video_id = extract_video_id(youtube_link)
        if video_id:
            st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

    if st.button("Process Video Transcript"):
        transcript_text = extract_transcript_details(youtube_link)
        if "Error" not in transcript_text and "Transcript is not available" not in transcript_text:
            with st.spinner("⏳ Processing transcript for Q&A..."):
                process_status = process_text_for_chat(transcript_text, "faiss_youtube")
                st.success(process_status)
        else:
            st.error(transcript_text)

    user_question = st.text_input("Ask a question about the video:")
    if user_question:
        response = get_answer(user_question, "faiss_youtube")
        st.markdown(f"🧑‍💻 You:** {user_question}")
        st.markdown(f"🤖 AI:** {response}")

# Local video chat
elif mode == "🎬 Chat with Local Video":
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"], help="Upload a video for AI analysis")
    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(video_file.read())
            video_path = temp_video.name

        st.video(video_path, format="video/mp4", start_time=0)

        user_query = st.text_area("What insights are you seeking from the video?", placeholder="Ask anything about the video content.")
        if st.button("🔍 Analyze Video"):
            if not user_query:
                st.warning("Please enter a question or insight to analyze the video.")
            else:
                try:
                    with st.spinner("Processing video and gathering insights..."):
                        processed_video = upload_file(video_path)
                        while processed_video.state.name == "PROCESSING":
                            time.sleep(1)
                            processed_video = get_file(processed_video.name)

                        analysis_prompt = f"Analyze the uploaded video for content and context. Respond to the following query using video insights and supplementary web research: {user_query}"
                        response = multimodal_Agent.run(analysis_prompt, videos=[processed_video])

                    st.subheader("Analysis Result")
                    st.markdown(response.content)
                except Exception as error:
                    st.error(f"An error occurred during analysis: {error}")
                finally:
                    Path(video_path).unlink(missing_ok=True)

st.markdown("<hr><p style='text-align: center;'>🚀 Built with ❤ using Streamlit & Gemini AI</p>", unsafe_allow_html=True)