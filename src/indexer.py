import faiss
import numpy as np

class FAISSIndex:
    def __init__(self, dimension=384):
        self.dimension = dimension
        try:
            faiss.omp_set_num_threads(1)
        except:
            pass
        # Flat Inner Product Index (acts as Cosine Similarity when vectors are L2-normalized)
        self.index = faiss.IndexFlatIP(dimension)
        
    def add(self, embeddings):
        """Adds normalized embeddings to the index."""
        embeddings_clean = np.ascontiguousarray(embeddings).astype('float32')
        self.index.add(embeddings_clean)
        
    def save(self, path):
        """Saves index to local binary file."""
        faiss.write_index(self.index, path)
        
    def load(self, path):
        """Loads index from binary file."""
        self.index = faiss.read_index(path)
        
    def search(self, query_embedding, k=1000):
        """Queries the index for top K nearest neighbors. Returns distances and indices."""
        if len(query_embedding.shape) == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
            
        q_clean = np.ascontiguousarray(query_embedding).astype('float32')
        distances, indices = self.index.search(q_clean, k)
        return distances[0], indices[0]
        
    def size(self):
        return self.index.ntotal
