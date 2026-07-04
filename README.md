# Local NotebookLM Clone

This is a generic Local NotebookLM clone based on Retrieval-Augmented Generation (RAG). It provides a web interface where you can manage different knowledge bases, upload documents, and chat with an AI assistant powered by Gemini about those documents.

## Features
- **Multiple Notebooks**: Support switching between different topics or knowledge bases.
- **RAG Powered**: Automatically slices text into semantic chunks and creates vector embeddings.
- **AI Assistant**: Conversational AI assistant with access to context from the current notebook.
- **Voice Interactions**: Connect and interact with a voice assistant for real-time discussions.
- **Interactive Quizzes**: Built-in generic quiz engine to test your understanding of the knowledge base.

## Setup Instructions

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Key**:
   Create a `.env` file and set your Gemini API key:
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key
   ```

3. **Start the Server**:
   ```bash
   python app.py
   ```
   Or simply run `start.bat` on Windows.

4. **Access UI**:
   Open `http://localhost:8000` in your browser.
