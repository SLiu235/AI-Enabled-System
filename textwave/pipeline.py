import os
import numpy as np
from typing import List, Dict, Any, Optional
from modules.extraction.preprocessing import DocumentProcessing
from modules.extraction.embedding import Embedding
from modules.retrieval.indexing import FaissIndex
from modules.retrieval.search import FaissSearch
from modules.retrieval.reranker import Reranker
from modules.generator.question_answering import QA_Generator
os.environ["MISTRAL_API_KEY"] = "TGwZGwfx2CCPujKn7FEq6uNLh3nQZxLh"

class Pipeline:
    def __init__(self, 
                 embedding_model='all-MiniLM-L6-v2', 
                 index_type='brute_force',
                 mistral_api_key="MISTRAL_API_KEY",
                 reranker_type='cross_encoder',
                 reranker_model='cross-encoder/ms-marco-MiniLM-L-6-v2',
                 qa_temperature=0.3,
                 qa_model="mistral-small-latest"):
        
        self.document_processor = DocumentProcessing()
        self.embedding_model = Embedding(embedding_model)
        self.index = None
        self.searcher = None
        self.index_type = index_type
        self.result = None
        
        self.reranker = Reranker(
            type=reranker_type,
            cross_encoder_model_name=reranker_model
        )
        
        self.qa_generator = None
        if mistral_api_key:
            self.qa_generator = QA_Generator(
                api_key=os.environ[mistral_api_key],
                temperature=qa_temperature,
                generator_model=qa_model
            )

    def preprocess_corpus(self, 
                         corpus_directory: str,
                         chunking_strategy: str = 'sentence',
                         fixed_length: Optional[int] = None,
                         overlap_size: int = 2) -> Dict[str, Any]:
        """
        Preprocess corpus by chunking documents and creating embeddings.
        
        Args:
            corpus_directory (str): Path to directory containing corpus documents
            chunking_strategy (str): Strategy for chunking ('sentence' or 'fixed-length')
            fixed_length (int): Size of chunks when using fixed-length strategy
            overlap_size (int): Number of overlapping units between chunks
            
        Returns:
            Dict containing the processed corpus information
        """
        # Validate inputs
        if chunking_strategy not in ['sentence', 'fixed-length']:
            raise ValueError("chunking_strategy must be either 'sentence' or 'fixed-length'")
            
        if chunking_strategy == 'fixed-length' and fixed_length is None:
            raise ValueError("fixed_length must be specified when using 'fixed-length' chunking strategy")

        all_chunks = []
        chunk_metadata = []
        
        # Process each document in the corpus directory
        for filename in os.listdir(corpus_directory):
            if not filename.endswith('.clean'):
                continue
                
            file_path = os.path.join(corpus_directory, filename)
            
            # Apply chunking strategy
            chunks = self.document_processor.split_document(
                document_filename=file_path,
                chunking_strategy=chunking_strategy,
                chunk_size=fixed_length,
                overlap_size=overlap_size
            )
            
            # Store chunks and metadata
            all_chunks.extend(chunks)
            chunk_metadata.extend([{
                'source_document': filename,
                'chunk_index': idx,
                'text': chunk
            } for idx, chunk in enumerate(chunks)])
        
        # Create embeddings for all chunks
        embeddings = self.embedding_model.encode(all_chunks)
        
        # Initialize FAISS index and add embeddings
        self.index = FaissIndex(index_type=self.index_type)
        self.index.add_embeddings(embeddings, metadata=chunk_metadata)
        
        # Initialize searcher
        self.searcher = FaissSearch(self.index)
        
        # Store results
        self.result = {
            'chunks': all_chunks,
            'metadata': chunk_metadata,
            'embeddings': embeddings,
            'index': self.index
        }
        
        return self.result

    def search_neighbors(self, query, k=10):
        """
        Search k-nearest neighbors for a query
        
        Args:
            query (str): Input query text
            k (int): Number of neighbors to retrieve
            
        Returns:
            List[Dict]: List of retrieved documents with metadata
        """
        if self.index is None or self.searcher is None:
            raise ValueError("Pipeline not initialized. Run preprocess_corpus first.")
        
        # Encode query
        query_embedding = self.__encode(query) 
        
        # Search for nearest neighbors
        distances, indices, metadata = self.searcher.search(query_embedding, k=k)
        
        # Format results
        results = []
        for i in range(len(metadata)):
            results.append({
                'text': metadata[i]['text'],
                'source': metadata[i]['source_document'],
                'chunk_index': metadata[i]['chunk_index'],
                'distance': float(distances[0][i])
            })
        
        return results

    def generate_answer(self, query, k=10, rerank=True):
        """
        Generate an answer for a query using retrieved and optionally reranked context.
        
        Args:
            query (str): Input query text
            k (int): Number of neighbors to retrieve
            rerank (bool): Whether to apply reranking to retrieved documents
            
        Returns:
            Dict containing the answer and relevant metadata
        """
        if self.qa_generator is None:
            raise ValueError("QA Generator not initialized. Please provide Mistral API key during pipeline initialization.")
            
        # First retrieve relevant documents
        retrieved_results = self.search_neighbors(query, k=k)
        context_docs = [result['text'] for result in retrieved_results]
        
        # Apply reranking if requested
        if rerank:
            reranked_docs, reranked_indices, rerank_scores = self.reranker.rerank(
                query=query,
                context=context_docs,
                full_chuncked=self.result['chunks'] if self.result else None
            )
            
            # Update context and metadata with reranked results
            context_docs = reranked_docs
            reranked_metadata = [retrieved_results[idx] for idx in reranked_indices]
            for i, metadata in enumerate(reranked_metadata):
                metadata['rerank_score'] = rerank_scores[i]
            retrieved_results = reranked_metadata
        
        # Generate answer using the context
        answer = self.qa_generator.generate_answer(query, context_docs)
        
        return {
            'answer': answer,
            'context': context_docs,
            'metadata': retrieved_results
        }
    
    def __encode(self, query: str) -> np.ndarray:
        """
        Encode query text into embedding vector
        
        Args:
            query (str): Input query text
            
        Returns:
            np.ndarray: Embedding vector
        """
        return self.embedding_model.encode([query])

    def save_index(self, faiss_path: str, metadata_path: str):
        """
        Save the FAISS index and metadata to disk
        
        Args:
            faiss_path (str): Path to save FAISS index
            metadata_path (str): Path to save metadata
        """
        if self.index is None:
            raise ValueError("No index to save. Run preprocess_corpus first.")
        self.index.save(faiss_path, metadata_path)

    def load_index(self, faiss_path: str, metadata_path: str):
        """
        Load the FAISS index and metadata from disk
        
        Args:
            faiss_path (str): Path to load FAISS index from
            metadata_path (str): Path to load metadata from
        """
        if self.index is None:
            self.index = FaissIndex(index_type=self.index_type)
        self.index.load(faiss_path, metadata_path)
        self.searcher = FaissSearch(self.index)

    def display_chunks(self):
        """Helper function to display chunk information"""
        if self.result is None:
            raise ValueError("No chunks to display. Run preprocess_corpus first.")
            
        print("Number of chunks:", len(self.result['chunks']))
        print("\nChunks:")
        for i, chunk in enumerate(self.result['chunks']):
            print(f"\nChunk {i+1}:")
            print(chunk)
        print("\nEmbedding shape:", self.result['embeddings'].shape)
        print("-" * 80)

if __name__ == "__main__":
    query = "Who was Abraham Lincoln?"

    pipeline = Pipeline()
    
    result1 = pipeline.preprocess_corpus('storage/corpus', 
                                       chunking_strategy='sentence', 
                                       overlap_size=2)
    result = pipeline.generate_answer(query=query, k=10)
    print(result)

    result2 = pipeline.preprocess_corpus('storage/corpus', 
                                        chunking_strategy='fixed-length', 
                                        fixed_length=150, 
                                        overlap_size=1)
    
    result = pipeline.generate_answer(query=query, k=10)
    print(result)