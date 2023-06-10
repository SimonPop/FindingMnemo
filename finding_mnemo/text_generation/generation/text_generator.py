from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import LogitsProcessor, LogitsProcessorList

class SampledTemperatureProcessor(LogitsProcessor):
    def __init__(self, temperature=1):
        self.T = temperature

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        soft_scores = F.softmax(scores/self.T, dim=1)
        return soft_scores * torch.rand(scores.shape[1])

class TextGenerator():
    def __init__(self, model_type: str = "t5"):

        self.model_type = model_type
        
        if model_type == "t5":
            self.config = {
                "num_beams": 10,
                "num_return_sequences": 2,
                "no_repeat_ngram_size": 1,
                "remove_invalid_values": True,
                "max_new_tokens": 40,
                "temperature": 0.95
            }
            self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
            self.tokenizer = AutoTokenizer.from_pretrained("t5-base")

        elif model_type == "k2t":
            from keytotext import pipeline
            self.config = {
                "max_length": 1024,
                "num_beams": 20,
                "length_penalty": 0.01,
                "no_repeat_ngram_size": 3,
                "early_stopping": True,
            }
            self.model = pipeline("k2t-base")

    def generate(self, **kwargs) -> List[str]:
        if self.model_type == "t5":
            return self.generate_from_constraint(**kwargs)
        elif self.model_type == "k2t":
            return self.generate_from_keywords(**kwargs)

    def generate_from_constraint(self, keywords: List[str], prompt: str) -> List[str]:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        force_words_ids = self.tokenizer(keywords, add_special_tokens=False).input_ids

        outputs = self.model.generate(
            input_ids,
            force_words_ids=force_words_ids,
            logits_processor=LogitsProcessorList([SampledTemperatureProcessor(self.config['temperature'])]),
            **self.config
        )

        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    def generate_from_keywords(self, keywords: List[str]) -> List[str]:
        return self.model(keywords, **self.config)