import gensim.downloader
import numpy as np
from numpy.linalg import norm

class EmbeddingHeuristic():
    def __init__(self):
        self.model = gensim.downloader.load('glove-wiki-gigaword-50')

    def word_embedding(self, word: str) -> np.array:
        """Embeds a word into a vector."""
        return self.model[word]
        
    def similarity(self, a: str, b: str) -> float:
        """Computes cosine similarity for two words."""
        embedding_a = self.word_embedding(a)
        embedding_b = self.word_embedding(b)
        cos_sim = np.dot(embedding_a, embedding_b)/(norm(embedding_a)*norm(embedding_b))
        return cos_sim

    def distance(self, a: str, b: str) -> float:
        return 1 / self.similarity(a,b) + 1e-3

EMBEDDING_HEURISTIC = EmbeddingHeuristic()

if __name__ == "__main__":
    print("High", EMBEDDING_HEURISTIC.similarity("penguin", "dog"))
    print("Medium", EMBEDDING_HEURISTIC.similarity("penguin", "marine"))
    print("Low", EMBEDDING_HEURISTIC.similarity("penguin", "table"))