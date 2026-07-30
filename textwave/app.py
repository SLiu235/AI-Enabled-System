from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from typing import Optional
import os
import uvicorn
from pipeline import Pipeline

app = FastAPI(title="Question Answering System")

# Initialize the pipeline
pipeline = Pipeline(
    embedding_model='all-MiniLM-L6-v2',
    index_type='brute_force',
    mistral_api_key="MISTRAL_API_KEY",
    reranker_type='tfidf_corpus',
    qa_temperature=0.3
)

@app.post("/process_corpus")
async def process_corpus(
    chunking_strategy: str = Form(default='fixed-length'),
    fixed_length: Optional[int] = Form(default=50),
    overlap_size: int = Form(default=10),
):
    """Process and index the corpus documents."""
    try:
        corpus_directory = "storage/corpus"
        if not os.path.exists(corpus_directory):
            os.makedirs(corpus_directory)
        
        result = pipeline.preprocess_corpus(
            corpus_directory=corpus_directory,
            chunking_strategy=chunking_strategy,
            fixed_length=fixed_length,
            overlap_size=overlap_size,
        )
        
        # Save the index
        pipeline.save_index(
            faiss_path="storage/index/faiss.index",
            metadata_path="storage/index/metadata.pkl"
        )
        
        return {"message": "Corpus processed successfully", "chunks": len(result['chunks'])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    """Upload a new document to the corpus."""
    try:
        corpus_directory = "storage/corpus"
        if not os.path.exists(corpus_directory):
            os.makedirs(corpus_directory)
        
        file_path = os.path.join(corpus_directory, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return {"message": f"Document {file.filename} uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(
    question: str = Form(...),
    k: Optional[int] = Form(default=5),
    rerank: Optional[bool] = Form(default=True)
):
    """Generate an answer for the given question."""
    try:
        if not os.path.exists("storage/index/faiss.index"):
            raise HTTPException(
                status_code=400, 
                detail="Index not found. Please process corpus first."
            )
        
        # Load the index if not already loaded
        if pipeline.index is None:
            pipeline.load_index(
                faiss_path="storage/index/faiss.index",
                metadata_path="storage/index/metadata.pkl"
            )
        
        # Generate answer
        result = pipeline.generate_answer(
            query=question,
            k=k,
            rerank=rerank
        )
        
        return {
            "question": question,
            "answer": result['answer'],
            "context": result['context'],
            "metadata": result['metadata']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
