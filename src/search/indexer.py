from jina import Document, DocumentArray, Executor
from pathlib import Path
import pandas as pd 
import torch

from src.model.sound_siamese_v2 import SoundSiamese

class Indexer(Executor):
    model: SoundSiamese

    def index(self) -> DocumentArray:
        model = self.load_model()
        da = self.load_documents()
        return da

    def load_model(self) -> SoundSiamese:
        model = SoundSiamese()
        model.load_state_dict(torch.load(Path(__file__).parent.parent / "model" / "model_dict"))
        self.model = model
        return model

    def load_documents(self) -> DocumentArray:
        # TODO: Use batches to encode faster.
        # TODO: phoneme + text
        dataframe = pd.read_csv(Path(__file__).parent.parent / 'dataset' / 'pairing' / 'english.csv')
        words = dataframe[['word', 'ipa']].astype(str).head(50)   
        # embedding = self.model.encode(words)
        with DocumentArray(
            storage='redis',
            config={
                'n_dim': self.model.embedding_dim,
                'index_name': 'english_words',
                'distance': 'COSINE'
            },
        ) as da:
            da.extend([Document(text=w['word'], embedding=self.model.encode([w['ipa']]).detach()[0]) for _, w in words.iterrows()])
        return da

    def embed(self, da: DocumentArray):
        da.embed(self.model.encode)


indexer = Indexer()
da = indexer.index()

np_query = indexer.model.encode(["bɔ́təl"]).detach().numpy()[0]
print(da.find(np_query, limit=5)[:, 'text'])