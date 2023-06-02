from pathlib import Path
from typing import Dict, Union

import torch
from docarray import Document, DocumentArray

from finding_mnemo.pairing.model.phonetic_siamese import PhoneticSiamese


class Engine():
    n_limit: int = 5
    model: PhoneticSiamese
    da: DocumentArray

    def __init__(self, documents):
        self.model = self.load_model()
        self.da = documents

    def search(self, docs: DocumentArray, **kwargs) -> Union[DocumentArray, Dict, None]:
        x = docs[:, "tags__ipa"]
        docs.embeddings = self.model.encode(x).detach()
        docs.match(self.da, metric="euclidean", limit=self.n_limit)
        return docs

    def load_model(self) -> PhoneticSiamese:
        model = PhoneticSiamese()
        model.load_state_dict(
            torch.load(Path(__file__).parent.parent / "model" / "model_dict")
        )
        model.eval()
        self.model = model
        return model