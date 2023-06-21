from pathlib import Path
from typing import Dict, Union, List

import json
import torch
from docarray import Document, DocumentArray

from finding_mnemo.pairing.model.phonetic_siamese import PhoneticSiamese
from finding_mnemo.pairing.utils.ipa import mandarin_ipa_to_en
from finding_mnemo.pairing.utils.distance import levenshtein_distance


class Engine():
    n_limit: int = 5
    model: PhoneticSiamese
    da: DocumentArray

    def __init__(self, documents, n_limit: int = 5):
        self.n_limit = n_limit
        self.model = self.load_model()
        self.da = documents

    def search(self, docs: DocumentArray, **kwargs) -> Union[DocumentArray, Dict, None]:
        x = docs[:, "tags__ipa"]
        x = [mandarin_ipa_to_en(w) for w in x]
        docs.embeddings = self.model.encode(x).detach()
        docs.match(self.da, metric="euclidean", limit=self.n_limit)
        return docs
    
    def get_distance(self, docs: DocumentArray, target_ipa: str) -> List[float]:
        docs_dict = docs.to_dict()
        ipas = [x['tags']['ipa'] for x in docs_dict[0]["matches"]]
        distances = [levenshtein_distance(target_ipa, ipa) for ipa in ipas]
        return distances

    def load_model(self) -> PhoneticSiamese:
        with open(Path(__file__).parent.parent / "model" / "model_config.json", "r") as f:
            model_config = json.load(f)
        model = PhoneticSiamese(
            embedding_dim=model_config["embedding_dim"],
            dim_feedforward=model_config["dim_feedforward"],
            nhead=model_config["nhead"],
            dropout=model_config["dropout"],
            batch_size=model_config["batch_size"],
            weight_decay=model_config["weight_decay"],
            lr=model_config["lr"],
            margin=model_config["margin"],
            lambda_triplet=model_config["lambda_triplet"],
            lambda_pos=model_config["lambda_pos"],
            lambda_neg=model_config["lambda_neg"],
        )
        model.load_state_dict(
            torch.load(Path(__file__).parent.parent / "model" / "model_dict")
        )
        self.model = model
        model.eval()
        return model