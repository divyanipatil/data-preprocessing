import torch
from torchtext.data.utils import get_tokenizer
import string
import spacy


class TextPreprocessor:
    def __init__(self):
        self.tokenizer = get_tokenizer("basic_english")
        # Load English language model for spaCy
        self.nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def to_lowercase(text):
        return text.lower()

    @staticmethod
    def remove_punctuation(text):
        return ''.join(char for char in text if char not in string.punctuation)

    def lemmatize_text(self, text):
        """
        Reduce words to their base/dictionary form
        Example: 'running' -> 'run', 'better' -> 'good', 'was' -> 'be'
        """
        doc = self.nlp(text)
        return ' '.join([token.lemma_ for token in doc])
