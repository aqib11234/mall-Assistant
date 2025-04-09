<h1 align="center">🎙️ Millennium Mall Voice Assistant</h1>

<p align="center">
  A powerful voice-enabled assistant that helps users navigate the  Mall by answering floor/store-related queries using audio input and AI-powered responses.
</p>

<hr>

<h2>🧠 Features</h2>
<ul>
  <li>🎤 <strong>Voice Input:</strong> Users can speak their questions directly into the app.</li>
  <li>📝 <strong>Whisper Speech-to-Text:</strong> Transcribes user audio into text using OpenAI’s Whisper model.</li>
  <li>📚 <strong>Contextual Retrieval:</strong> Uses LangChain + FAISS to search through mall data extracted from a PDF.</li>
  <li>💬 <strong>LLM-Powered Responses:</strong> Uses Llama3.2 via Ollama to generate natural language answers based on your question.</li>
  <li>🔊 <strong>Voice Response:</strong> Converts the LLM output to speech using gTTS (Google Text-to-Speech).</li>
</ul>

<h2>⚙️ Installation</h2>

<pre><code>git clone https://github.com/aqib11234/Mall_Assistant.git
cd Mall_Assistant
python -m venv venv
venv\Scripts\activate   # On Windows
pip install -r requirements.txt
streamlit run app.py    # Or whatever your main file is named
</code></pre>

<h2>📁 Project Structure</h2>

<pre><code>.
├── app.py                 # Main Streamlit application
├── data.pdf              # Mall information document (used for RAG)
├── faiss_index/          # Saved FAISS vector store
├── requirements.txt      # Python dependencies
└── README.md             # You are here!
</code></pre>

<h2>📦 Requirements</h2>
<ul>
  <li>Python 3.9+</li>
  <li>Streamlit</li>
  <li>LangChain</li>
  <li>Whisper & Transformers</li>
  <li>FAISS</li>
  <li>gTTS</li>
</ul>

<h2>💡 Usage</h2>
<ol>
  <li>Click <strong>Start Recording</strong> and speak your question about the mall.</li>
  <li>Click <strong>Stop Recording</strong> once done.</li>
  <li>The assistant will transcribe, fetch relevant info, and respond back via text and audio.</li>
</ol>

<h2>🚀 Technologies Used</h2>
<ul>
  <li>Streamlit</li>
  <li>OpenAI Whisper</li>
  <li>LangChain</li>
  <li>FAISS</li>
  <li>gTTS</li>
  <li>HuggingFace Transformers</li>
</ul>


<h2>🙌 Acknowledgements</h2>
<ul>
  <li><a href="https://ollama.com">Ollama</a></li>
  <li><a href="https://huggingface.co">Hugging Face</a></li>
  <li><a href="https://streamlit.io">Streamlit</a></li>
</ul>
