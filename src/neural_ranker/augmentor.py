import random
import nltk
from nltk.corpus import wordnet
import re

# Ensure WordNet is downloaded
nltk.download('wordnet', quiet=True)

class TextAugmentor:
    """
    A more advanced text augmentation class that combines:
    - Word dropout: randomly dropping words.
    - Word swap: swapping adjacent words.
    - Synonym replacement: replacing a word with one of its synonyms.
    - Sentence shuffling (optional): shuffling the order of sentences.
    """
    def __init__(self, dropout_prob=0.1, swap_prob=0.1, synonym_prob=0.1, shuffle_sentences=False):
        self.dropout_prob = dropout_prob
        self.swap_prob = swap_prob
        self.synonym_prob = synonym_prob
        self.shuffle_sentences = shuffle_sentences

    def augment(self, text: str) -> str:
        # Optionally shuffle sentence order
        if self.shuffle_sentences:
            sentences = re.split(r'(?<=[.!?]) +', text)
            if len(sentences) > 1:
                random.shuffle(sentences)
                text = " ".join(sentences)
        
        words = text.split()
        augmented_words = []
        i = 0
        while i < len(words):
            word = words[i]
            rand_val = random.random()
            
            # Word dropout: drop the word if rand_val is less than dropout probability.
            if rand_val < self.dropout_prob:
                i += 1
                continue
            
            # Word swap: swap current word with the next word if possible.
            elif rand_val < self.dropout_prob + self.swap_prob and i < len(words) - 1:
                # Swap current and next word
                augmented_words.append(words[i+1])
                augmented_words.append(word)
                i += 2  # Skip next word since it's been swapped
                continue
            
            # Synonym replacement: attempt to replace with a synonym.
            elif rand_val < self.dropout_prob + self.swap_prob + self.synonym_prob:
                synonym = self.get_synonym(word)
                if synonym:
                    augmented_words.append(synonym)
                else:
                    augmented_words.append(word)
            else:
                augmented_words.append(word)
            i += 1

        # Fallback to original if all words were dropped
        if len(augmented_words) == 0:
            augmented_words = words
        return " ".join(augmented_words)

    def get_synonym(self, word: str) -> str:
        """
        Retrieve a synonym for the given word from WordNet.
        If no synonym is found, returns None.
        """
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.append(synonym)
        if synonyms:
            return random.choice(synonyms)
        else:
            return None