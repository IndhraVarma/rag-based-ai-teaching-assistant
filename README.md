# RAG-Based AI Teaching Assistant 🎓🤖

This project implements a **Retrieval-Augmented Generation (RAG) based AI Teaching Assistant** capable of ingesting videos and audio, transcribing them, chunking and embedding the content, and generating intelligent, context-aware responses to user questions.

The system is designed to act like a **personal AI tutor**, grounding its answers strictly in the provided learning material rather than relying on generic knowledge.

---

## 📌 Project Overview

Traditional LLMs can hallucinate or provide generic answers. This project solves that by combining:

* **Speech-to-Text (STT)** for lecture ingestion
* **Vector embeddings** for semantic search
* **Retrieval-Augmented Generation (RAG)** for accurate, source-based answers

The assistant retrieves the most relevant content chunks and uses them to generate precise educational responses.

---

## 🧠 System Architecture (High-Level Flow)

1. **Input Content** (videos / audio)
2. **Speech-to-Text Transcription**
3. **Text Chunking**
4. **Embedding Generation & Storage**
5. **Semantic Retrieval**
6. **LLM-Based Answer Generation**

---

## 📂 Project Structure

```
├── __pycache__/              # Python cache files
├── audios/                   # Extracted or uploaded audio files
├── Videos/                   # Input video files (lectures/tutorials)
├── outputs/                  # Generated transcripts and responses
├── whisper/                  # Whisper-based speech-to-text utilities
│
├── embeddings.joblib         # Stored vector embeddings
├── process_video.py          # Video ingestion and audio extraction
├── process_incoming.py       # Handles new content ingestion pipeline
├── stt.py                    # Speech-to-text processing
├── read_chunks.py            # Text chunking logic
├── prompt.txt                # Prompt template for LLM responses
├── response.py               # RAG-based response generation
├── tempCodeRunnerFile.py     # Temporary runtime file
└── README.md                 # Project documentation
```

---

## 🔧 Technologies & Tools Used

* **Python**
* **Whisper** – speech-to-text transcription
* **LangChain-style RAG pipeline** (custom implementation)
* **Vector embeddings** (stored via `joblib`)
* **LLM (API-based or local)** – response generation
* **FFmpeg** – audio extraction from video

---

## ⚙️ Core Components Explained

### 🎥 Video & Audio Processing

* Extracts audio from video files
* Supports multiple input formats

### 🗣️ Speech-to-Text (STT)

* Converts lecture audio into clean text transcripts
* Stores outputs for traceability

### ✂️ Chunking & Embeddings

* Splits transcripts into semantically meaningful chunks
* Converts chunks into vector embeddings
* Saves embeddings for fast retrieval

### 🔍 Retrieval-Augmented Generation (RAG)

* Retrieves top relevant chunks based on user query
* Injects retrieved context into the prompt
* Generates grounded, accurate answers

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Add Input Videos or Audio

* Place files inside the `Videos/` or `audios/` directory

### 3️⃣ Process Incoming Content

```bash
python process_incoming.py
```

### 4️⃣ Ask Questions

* Run the response pipeline:

```bash
python response.py
```

* Enter a query related to the ingested content

---

## 🧪 Example Use Cases

* AI teaching assistant for recorded lectures
* Corporate training knowledge assistant
* Personalized learning companion
* Course-specific Q&A bot

---

## 💡 Future Improvements

* Web UI (Streamlit / React)
* Support for PDFs and slides
* Source citation in answers
* Hybrid search (keyword + vector)
* Incremental embedding updates

---

## 👤 Author

Built as a **RAG-based AI Teaching Assistant** project for educational and research purposes.

---

⭐ If this project helped you, consider starring the repository and extending it!
