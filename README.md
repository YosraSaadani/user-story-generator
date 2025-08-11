# RAG-JIRA Story Generator

A comprehensive Retrieval-Augmented Generation (RAG) system designed to process documents containing software requirements, classify and chunk content semantically, retrieve relevant context, generate structured Agile artifacts (user stories, epics, acceptance criteria), and integrate with JIRA for story creation. Built with LangChain, ChromaDB, Ollama, FastAPI, and Hugging Face models.

## Overview

This project automates the transformation of raw requirement documents (TXT, PDF, DOCX) into actionable JIRA stories. It uses advanced NLP for classification, semantic chunking for optimal retrieval, vector embeddings for similarity search, and local LLMs (via Ollama) for generation. The system includes a FastAPI backend for uploading documents, generating summaries, and pushing to JIRA.

Key components:
- **Document Processing**: Loads, parses, chunks, and classifies requirements.
- **Vector Store**: ChromaDB for storing embeddings of processed chunks.
- **Retrieval**: Semantic search with filters for context retrieval.
- **Generation**: Ollama-based query engine to create user stories and epics.
- **Integration**: Pushes generated stories to JIRA via API.
- **API**: FastAPI endpoints for upload, download, and JIRA push.

## Features

- Multi-format document support (TXT, PDF, DOCX, JSON datasets).
- Zero-shot classification of requirements (e.g., user stories, acceptance criteria).
- Semantic chunking with configurable size and overlap.
- RAG pipeline with Ollama (Mistral model) for generating structured outputs.
- JSONL output for generated stories, convertible to readable DOCX.
- JIRA integration for creating issues with custom fields (e.g., sprint, epic).
- Interactive CLI for querying the RAG system.
- Batch processing and statistics for chunks and vectors.
- CORS-enabled FastAPI for frontend integration.

## Project Structure

```
MAIN PROJECT/
├── .alpackages/               # Auto-generated packages
├── .snapshots/                # Version control snapshots
├── chroma_db/                 # Chroma vector database storage
│   └── ffdf69ec-10c9-48a3-b788-747ae2c8458f/  # Collection data
├── Fine tuning/               # Fine-tuning scripts/models (if applicable)
├── LLM-datasets/              # Datasets for LLM training
│   └── SEPERATE DATASETS/     # Individual dataset files
├── outputs/                   # Generated files (JSONL, DOCX)
├── processed_data/            # Processed JSON datasets
├── raw_data/                  # Raw input datasets (JSON)
├── test_inputs/               # Test documents for processing
├── __pycache__/               # Python cache
├── api.py                     # FastAPI application
├── chunking_processor.py      # Document chunking logic
├── classifier.py              # Zero-shot text classification
├── code_engine.py             # Ollama query engine for RAG
├── context_retriever.py       # Context retrieval from vector store
├── data_processing.py         # Raw data to structured JSON processor
├── document_loader.py         # LangChain document loader
├── document_processor.py      # Smart parsing and semantic chunking
├── get_jira_by_id.py          # JIRA user info fetcher
├── jsonl_to_readable_text.py  # Convert JSONL to DOCX summary
├── main_rag.py                # Main RAG system orchestration
├── models.py                  # Data models (e.g., DocumentChunk)
├── processing_script.py       # Dataset processing script
├── push_stories_to_jira.py    # Push generated stories to JIRA
├── rag_processor.py           # Full RAG processing pipeline
├── setup_and_run.py           # Setup and interactive run script
├── vector_store.py            # ChromaDB vector store manager
├── requirements.txt           # Dependencies (create if needed)
├── .env                       # Environment variables (JIRA config)
└── README.md                  # This file
```

## Installation

1. **Clone the Repository**:
   ```
   git clone <repository-url>
   cd MAIN PROJECT
   ```

2. **Set Up Virtual Environment**:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Create a `requirements.txt` with:
   ```
   fastapi
   uvicorn
   langchain
   langchain-community
   sentence-transformers
   chromadb
   transformers
   pypdf2
   python-docx
   nltk
   spacy
   requests
   python-dotenv
   tqdm
   numpy
   scikit-learn
   ```
   Then run:
   ```
   pip install -r requirements.txt
   ```
   Download spaCy model:
   ```
   python -m spacy download en_core_web_sm
   ```
   Download NLTK data:
   ```
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

4. **Install and Run Ollama**:
   - Download Ollama: [ollama.ai](https://ollama.ai)
   - Make sure you have Mistral pulled and downloaded from Ollama:
     ```
     ollama pull mistral
     ollama serve
     ```
   - Ensure Ollama server is running on `http://localhost:11434`

5. **Configure Environment**:
   Create `.env` file with the following placeholders (update with your own JIRA details before running):
   ```
   JIRA_EMAIL=your-email@example.com
   JIRA_API_TOKEN=your-api-token
   JIRA_BASE_URL=https://your-domain.atlassian.net
   JIRA_PROJECT_KEY=YOURPROJECTKEY
   URL=https://your-domain.atlassian.net/rest/api/3/issue
   ```
   Before running any scripts that interact with JIRA, update the `.env` file with your own JIRA account details. This should be easy and quick.

## Usage

### 1. Process Datasets
Run the dataset processor to convert raw JSON to structured training data:
```
python processing_script.py
```
This generates `processed_data/training_data.json`.

If you encounter any errors along the way (mostly due to undownloaded packages), run:
```
pip install -r requirements.txt
```

### 2. Build Vector Database
Setup the RAG system (creates/loads ChromaDB):
```
python setup_and_run.py
```
Use `--recreate-db` flag to force rebuild.

### 3. Interactive CLI
Start interactive query session:
```
python main_rag.py
```
- Type questions (e.g., "Convert this requirement into user stories").
- Commands: `exit`, `help`, `stats`.

### 4. Run API Server
In the project directory, activate the FastAPI server:
```
uvicorn api:app --reload
```
Open the UI file (or access via browser at `http://localhost:8000/docs` for Swagger UI) and test your project.

Endpoints:
- **POST /upload_and_generate/**: Upload document → Process with RAG → Generate JSONL/DOCX.
  - Body: `file` (UploadFile)
  - Response: `{ "download_link": "/download_docx" }`
- **GET /download_docx**: Download generated DOCX summary.
- **POST /push_to_jira**: Push JSONL stories to JIRA.
  - Response: `{ "message": "Pushed to Jira", "download_link": "/download_docx" }`

### 5. Process a Single File
Use `rag_processor.py` for standalone processing:
```
python rag_processor.py
```
Processes `test_inputs/test2.pdf` (configurable).

### 6. Push to JIRA
After generation:
```
python push_stories_to_jira.py
```
Uses `outputs/generated_rag_output.jsonl`.

## Configuration Options

- **Chunking**: Adjust `chunk_size` (1000) and `chunk_overlap` (200) in scripts.
- **Embeddings**: Change model in `vector_store.py` (default: all-MiniLM-L6-v2).
- **Ollama**: Set model/URL in `code_engine.py`.
- **JIRA**: Custom fields in `push_stories_to_jira.py`.

## Dependencies

- Python 3.8+
- See `requirements.txt` for full list.

## Troubleshooting

- **Ollama Not Running**: Check `http://localhost:11434/api/tags`.
- **Vector Store Issues**: Delete `chroma_db/` and recreate.
- **JIRA Errors**: Verify API token and account ID in `.env`.
- **Model Loading**: Ensure GPU/CPU resources for embeddings/LLM.

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Commit changes.
4. Push and open a Pull Request.

## License

MIT License. See [LICENSE](LICENSE) for details.
