import torch
import random
from torchtext.vocab import GloVe
from typing import List


class TextAugmenter:
    def __init__(self):
        # Load GloVe embeddings
        self.glove = GloVe(name='6B', dim=100)
        # Cache for similar words to avoid repeated calculations
        self.similar_words_cache = {}

    def find_similar_words(self, word: str, n: int = 5) -> List[str]:
        """Find similar words using GloVe embeddings"""
        # Check cache first
        if word in self.similar_words_cache:
            return self.similar_words_cache[word]

        # Get word vector
        word_vector = self.glove[word]
        if word_vector is None:
            return []

        # Calculate cosine similarity with all words
        cos = torch.nn.CosineSimilarity(dim=0)
        distances = []

        # Only compare with first 50000 words for efficiency
        for similar_word in self.glove.itos[:50000]:
            if similar_word == word:
                continue
            similarity = cos(word_vector, self.glove[similar_word])
            distances.append((similar_word, similarity))

        # Sort by similarity and get top n
        similar_words = sorted(distances, key=lambda x: x[1], reverse=True)[:n]
        result = [word for word, _ in similar_words]

        # Cache the result
        self.similar_words_cache[word] = result
        return result

    def word_swap(self, text: str, swap_percent: float = 0.1) -> str:
        """Randomly swap adjacent words"""
        # Split text into sentences
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return text

        result_sentences = []
        for sentence in sentences:
            words = sentence.split()
            if len(words) >= 2:
                # Determine number of swaps for this sentence
                n_swaps = max(1, int(len(words) * swap_percent))

                for _ in range(n_swaps):
                    # Pick a random position (excluding last word)
                    idx = random.randint(0, len(words) - 2)
                    # Swap with next word
                    words[idx], words[idx + 1] = words[idx + 1], words[idx]

            result_sentences.append(' '.join(words))

        return '. '.join(result_sentences) + ('.' if text.endswith('.') else '')

    def synonym_replacement(self, text: str, replace_percent: float = 0.1) -> str:
        """Replace random words with their similar words from GloVe embeddings"""
        # Split text into sentences
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return text

        result_sentences = []
        for sentence in sentences:
            words = sentence.split()
            if words:
                # Determine number of replacements for this sentence
                n_replacements = max(1, int(len(words) * replace_percent))

                # Choose random words to replace
                replace_indices = random.sample(range(len(words)), min(n_replacements, len(words)))

                for idx in replace_indices:
                    word = words[idx].lower()
                    # Get similar words
                    similar_words = self.find_similar_words(word)

                    # Replace word with random similar word if available
                    if similar_words:
                        words[idx] = random.choice(similar_words)

            result_sentences.append(' '.join(words))

        return '. '.join(result_sentences) + ('.' if text.endswith('.') else '')
