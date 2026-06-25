import os
from sentence_transformers import SentenceTransformer

class CandidateEmbedder:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5", cache_dir=None):
        # We enforce loading onto CPU to strictly respect sandbox constraints
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir, device="cpu")
        
    def embed_texts(self, texts, batch_size=256, show_progress=False):
        """Computes dense vectors for a list of texts. Embeddings are normalized to unit length (L2)."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True
        )
