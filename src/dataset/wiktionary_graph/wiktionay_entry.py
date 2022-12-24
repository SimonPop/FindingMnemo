from typing import List
from pydantic import BaseModel

class WiktionaryEntry(BaseModel):
    title: str # Used as label
    definition: str # Later used to get an encoding.
    related_words: List[str] # Used to find neighbors in the graph