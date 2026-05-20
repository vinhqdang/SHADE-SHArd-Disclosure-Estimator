"""Embedding cache and batching utilities."""
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

class DocumentEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", cache_dir: str = "data"):
        """
        Initialize the document embedder with a sentence transformer model.
        
        Args:
            model_name: HuggingFace model string. Default is bge-large-en-v1.5 as per plan.
            cache_dir: Base directory to store embedding caches.
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None
        
    @property
    def model(self):
        """Lazy load the model to save memory if only loading cache."""
        if self._model is None:
            print(f"Loading embedding model {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_corpus(self, corpus: List[str], dataset_name: str, batch_size: int = 32) -> np.ndarray:
        """
        Embed a list of documents, with disk caching to avoid recomputation.
        
        Args:
            corpus: List of document strings.
            dataset_name: Name for cache file (e.g. 'medqa', 'legalbench').
            batch_size: Batch size for model inference.
            
        Returns:
            np.ndarray of shape (num_docs, embedding_dim)
        """
        dataset_dir = os.path.join(self.cache_dir, dataset_name.lower())
        os.makedirs(dataset_dir, exist_ok=True)
        cache_file = os.path.join(dataset_dir, "embeddings.npy")
        
        if os.path.exists(cache_file):
            print(f"Loading cached embeddings from {cache_file}")
            embeddings = np.load(cache_file)
            if embeddings.shape[0] == len(corpus):
                return embeddings
            else:
                print(f"Cache size mismatch (found {embeddings.shape[0]}, expected {len(corpus)}). Recomputing...")
                
        print(f"Computing embeddings for {len(corpus)} documents...")
        embeddings = self.model.encode(
            corpus, 
            batch_size=batch_size, 
            show_progress_bar=True, 
            normalize_embeddings=True
        )
        
        print(f"Saving embeddings to {cache_file}")
        np.save(cache_file, embeddings)
        
        return embeddings
