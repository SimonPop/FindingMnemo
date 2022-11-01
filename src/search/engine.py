from jina import Document, DocumentArray, Executor
from pathlib import Path
import pandas as pd 
import torch
from src.model.sound_siamese_v2 import SoundSiamese

class Engine:
    # TODO: use executor
    n_limit: int = 5
    model: SoundSiamese
    da: DocumentArray

    def __init__(self):
        self.model = self.load_model()
        self.da = self.documents()

    def find(self, word: str):
        np_query = self.model.encode([word]).detach().numpy()[0]
        print(np_query)
        return self.da.find(np_query, limit=self.n_limit)

    def load_model(self) -> SoundSiamese:
        model = SoundSiamese()
        model.load_state_dict(torch.load(Path(__file__).parent.parent / "model" / "model_dict"))
        self.model = model
        return model

    def documents(self) -> DocumentArray:
        return DocumentArray(
            storage='redis',
            config={
                'n_dim': self.model.embedding_dim,
                'index_name': 'english_words',
                'distance': 'COSINE'
            },
        )