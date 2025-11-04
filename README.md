```markdown
# PDF RAG System

A Retrieval-Augmented Generation (RAG) system that allows you to query your PDF documents using local LLMs via Ollama.

## Features

- **PDF Ingestion**: Automatically processes all PDFs in a folder
- **Smart Chunking**: Splits documents into manageable chunks for better retrieval
- **Local LLM Integration**: Uses Ollama with TinyLlama for private, offline queries
- **SQLite Storage**: Efficient vector-like storage using cosine similarity
- **Interactive CLI**: Simple command-line interface for easy use

## Prerequisites

- [Go](https://golang.org/dl/) 1.21 or later
- [Ollama](https://ollama.ai/) installed and running
- TinyLlama model pulled in Ollama

### Install Ollama and TinyLlama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# In another terminal, pull TinyLlama
ollama pull tinyllama
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/DomiGames/RAG
cd RAG
```

2. Install dependencies:
```bash
go get github.com/mattn/go-sqlite3
go get github.com/ollama/ollama/api
go get github.com/ledongthuc/pdf
```

## Usage

1. **Prepare your PDFs**: Place all your PDF files in the `documents/` folder

2. **Run the application**:
```bash
go run main.go
```

3. **Available commands**:
   - `ingest` - Process all PDFs in the documents folder
   - `ingest /path/to/folder` - Process PDFs in a specific folder
   - `ingest /path/to/file.pdf` - Process a single PDF file
   - `status` - Show ingestion status and chunk counts
   - `clear` - Clear the database and start fresh
   - `your question` - Ask a question about your documents

### Example Session

```bash
> ingest
Scanning folder: documents
Found 3 PDF files
[1/3] Ingesting: documents/book1.pdf
Saved 150 chunks from documents/book1.pdf
[2/3] Ingesting: documents/book2.pdf
Saved 85 chunks from documents/book2.pdf
[3/3] Ingesting: documents/book3.pdf
Saved 42 chunks from documents/book3.pdf
Finished: 3/3 PDFs successfully ingested from: documents

> status
Current Database Status:
-----------------------
book1.pdf: 150 chunks
book2.pdf: 85 chunks
book3.pdf: 42 chunks
-----------------------
Total: 3 files, 277 chunks

> What is machine learning?
AI: Based on the context from the ingested documents, machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed...
```

## How It Works

1. **PDF Processing**: Extracts text from PDF files using the ledongthuc/pdf library
2. **Chunking**: Splits text into 200-word chunks for efficient retrieval
3. **Storage**: Stores chunks in SQLite database with source file tracking
4. **Retrieval**: Uses cosine similarity to find the most relevant chunks for a query
5. **Generation**: Sends the retrieved context and query to TinyLlama via Ollama for answer generation

## Project Structure

```
RAG/
├── main.go                 # Main application code
├── README.md              # This file
├── go.mod                 # Go module file
├── go.sum                 # Dependency checksums
├── documents/             # Put your PDFs here
│   ├── book1.pdf
│   ├── book2.pdf
│   └── ...
└── embeddings/
    └── store.db           # SQLite database (auto-created)
```

## Troubleshooting

### Common Issues

1. **"connection refused" errors**:
   - Ensure Ollama is running: `ollama serve`
   - Check if TinyLlama is installed: `ollama list`

2. **PDF parsing errors**:
   - Some PDFs may be corrupted or use non-standard encoding
   - The system will skip problematic files and continue with others

3. **Dependency installation issues**:
   ```bash
   # If you have network issues, try setting a Go proxy:
   go env -w GOPROXY=https://goproxy.io,direct
   ```

### Performance Tips

- For better results, use larger Ollama models (llama2, mistral, etc.)
- Adjust chunk size in code (currently 200 words) for your use case
- Clear database with `clear` command if you change chunking strategy

## Limitations

- Basic similarity matching (cosine similarity on word frequencies)
- No proper vector embeddings
- PDF text extraction may fail on some files
- Single-threaded processing


