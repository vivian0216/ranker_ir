import random
class TextAugmentor:
    """
    A simple text augmentation class.
    For example: synonym replacement, word dropout, shuffling sentences, etc.
    In this simple one, we randomly drop words with a given probability.
    """
    def __init__(self, dropout_prob=0.1):
        self.dropout_prob = dropout_prob

    def augment(self, text: str) -> str:
        words = text.split()
        # randomly drop words
        new_words = [word for word in words if random.random() > self.dropout_prob]
        # if all words are dropped, fallback to original text
        if len(new_words) == 0:
            new_words = words
        return " ".join(new_words)