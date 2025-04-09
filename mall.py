import os
import torch
import sounddevice as sd
import soundfile as sf
import tempfile
import streamlit as st
from gtts import gTTS
from PyPDF2 import PdfReader
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Initialize session state variables
if 'recording_state' not in st.session_state:
    st.session_state.recording_state = False
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'response_audio' not in st.session_state:
    st.session_state.response_audio = None

# Cleanup function for temporary files
def cleanup_temp_files():
    if st.session_state.audio_file and os.path.exists(st.session_state.audio_file):
        os.unlink(st.session_state.audio_file)
        st.session_state.audio_file = None
    
    if st.session_state.response_audio and os.path.exists(st.session_state.response_audio):
        os.unlink(st.session_state.response_audio)
        st.session_state.response_audio = None

# Initialize FAISS index for retrieval
@st.cache_resource
def initialize_faiss():
    pdf_path = "data.pdf"
    faiss_index_path = "faiss_index"

    if os.path.exists(faiss_index_path):
        st.write("Loading FAISS index from file...")
        model_name = "BAAI/bge-large-en-v1.5"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        model = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        document_search = FAISS.load_local(faiss_index_path, model, allow_dangerous_deserialization=True)
    else:
        st.write("Generating FAISS index and saving to file...")
        pdfreader = PdfReader(pdf_path)
        raw_text = ''.join([page.extract_text() for page in pdfreader.pages if page.extract_text()])
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
        texts = text_splitter.split_text(raw_text)

        model_name = "BAAI/bge-large-en-v1.5"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        model = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        document_search = FAISS.from_texts(texts, model)
        document_search.save_local(faiss_index_path)

    return document_search.as_retriever()

# Get a response from the LLM with context from FAISS
def get_llm_response_with_context(question, retriever):
    llm = ChatOllama(
        model="llama3.2:3b",
        temperature=0.1,
    )
    template = """You are Millennium mall assistant built to guide people by telling which stores are on which floor. Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    return response

# Convert text to speech
def text_to_speech(text):
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tts = gTTS(text=text, lang='en')
    tts.save(temp_file.name)
    return temp_file.name

# Initialize Whisper model and processor (cached to prevent reloading)
@st.cache_resource
def initialize_whisper():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe

# Record audio function with non-blocking stop functionality
def record_audio_with_stop():
    samplerate = 16000
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    filename = temp_file.name

    st.session_state.audio_data = []  # Store audio data in session state

    def callback(indata, frames, time, status):
        if st.session_state.recording_state:
            st.session_state.audio_data.extend(indata.copy())
        else:
            raise sd.CallbackStop  # Stop recording when recording_state is False

    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback, dtype="float32"):
        try:
            while st.session_state.recording_state:
                pass  # Keep the stream open
        except sd.CallbackStop:
            # Save recorded audio to file
            sf.write(filename, st.session_state.audio_data, samplerate)
            return filename

# Streamlit app
def main():
    st.title("Millennium Mall Assistant")
    st.write("Record your voice, and the assistant will guide you.")

    # Initialize models and retriever
    pipe = initialize_whisper()
    if st.session_state.retriever is None:
        st.session_state.retriever = initialize_faiss()

    # Start Recording Button
    if st.button("Start Recording", key="start_record_button"):
        st.session_state.recording_state = True  # Enable recording
        cleanup_temp_files()  # Clear previous files

        with st.spinner("Recording... Speak now!"):
            try:
                st.session_state.audio_file = record_audio_with_stop()
                st.success("Recording complete!")
            except Exception as e:
                st.error(f"An error occurred while recording: {e}")

    # Stop Recording Button
    if st.button("Stop Recording", key="stop_record_button"):
        st.session_state.recording_state = False  # Stop the recording

    # Process recorded audio if available
    if st.session_state.audio_file:
        st.subheader("Your Question (Audio):")
        st.audio(st.session_state.audio_file, format="audio/wav")

        # Transcribe and process
        with st.spinner("Processing the audio..."):
            audio_input, _ = sf.read(st.session_state.audio_file)
            transcription = pipe(audio_input)
            transcribed_text = transcription["text"]

            st.subheader("Transcribed Question:")
            st.write(transcribed_text)

            st.write("Fetching response from the assistant...")
            response = get_llm_response_with_context(transcribed_text, st.session_state.retriever)

            st.subheader("Assistant Response:")
            st.write(response)

            # Convert response to speech
            st.write("Generating voice response...")
            response_audio_file = text_to_speech(response)
            st.session_state.response_audio = response_audio_file

            st.subheader("Assistant Voice Response:")
            st.audio(response_audio_file, format="audio/mp3")


if __name__ == "__main__":
    main()
