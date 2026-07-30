# Textwave System 

## Overview
A FastAPI-based question answering system that processes documents, performs semantic search, and generates answers using advanced NLP techniques. The system uses sentence transformers for embeddings, FAISS for efficient similarity search, and Mistral AI for answer generation.


## Directory Structure
```
.
├── app.py                 
├── Dockerfile           
├── requirements.txt     
├── storage/            
│   ├── corpus/         
│   └── index/          
└── modules/           
    ├── extraction/    
    ├── retrieval/     
    └── generator/      
```

## Features
- Document upload and processing
- Flexible text chunking strategies
- Semantic search with FAISS indexing
- Cross-encoder reranking
- Question answering using Mistral AI

## Installation

### Local Setup
1. Clone the repository:
```bash
git clone https://github.com/creating-ai-enabled-systems-fall-2024/liu-sichao/textwave.git
cd textwave
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Mistral API key:
```bash
export MISTRAL_API_KEY="your-api-key-here"
```

### Docker Setup
1. Build the Docker image:
```bash
docker build -t textwave .
```

2. Run the container:
```bash
docker run -d \
    -p 8000:8000 \
    -v $(pwd)/storage:/app/storage \
    -e MISTRAL_API_KEY="your-api-key-here" \
    textwave
```

### API Endpoints

1. **Upload Document**
```bash
curl -X POST "http://localhost:8000/upload_document" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/document.pdf"
```

2. **Process Corpus**
```bash
curl -X POST "http://localhost:8000/process_corpus" \
     -H "accept: application/json" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "chunking_strategy=sentence&overlap_size=2"
```

3. **Ask Question**
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "accept: application/json" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "question=Your question here?&k=5&rerank=true"
```

### Web Interface
Access the interactive API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Configuration Options

### Corpus Processing
- `chunking_strategy`: Choose between 'sentence' or 'fixed_length'
- `fixed_length`: Number of tokens for fixed-length chunking
- `overlap_size`: Number of overlapping sentences/tokens between chunks

### Question Answering
- `k`: Number of similar documents to retrieve (default: 5)
- `rerank`: Enable/disable cross-encoder reranking (default: true)
