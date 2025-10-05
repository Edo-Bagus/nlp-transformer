import numpy as np
from collections import Counter
import re


class Tokenizer:
    """
    A simple word-level tokenizer for natural language text.

    This tokenizer builds a vocabulary from a list of training texts,
    assigns unique integer IDs to words that appear frequently enough,
    and provides encode/decode methods to map between text and token IDs.

    Attributes:
        vocab (dict[str, int]): Mapping from token string to token ID.
        reverse_vocab (dict[int, str]): Mapping from token ID back to token string.
        vocab_size (int): The total size of the vocabulary.
        special_tokens (dict[str, int]): Reserved special tokens:
            - "<PAD>": Padding token (0)
            - "<UNK>": Unknown token (1)
            - "<START>": Sequence start token (2)
            - "<END>": Sequence end token (3)
    """

    def __init__(self):
        """Initialize the tokenizer with special tokens and empty vocabulary."""
        self.special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<START>": 2,
            "<END>": 3
        }
        self.vocab = self.special_tokens.copy()
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def fit(self, texts, min_freq=2):
        """
        Build a vocabulary from a list of texts.

        Args:
            texts (list[str]): List of sentences or documents used for training.
            min_freq (int, optional): Minimum frequency for a word to be added to the vocabulary.
                Defaults to 2.

        Notes:
            - All words are lowercased.
            - Punctuation is removed.
            - Words appearing fewer than `min_freq` times are ignored.
            - Special tokens are always included.

        Example:
            >>> tokenizer = Tokenizer()
            >>> tokenizer.fit(["hello world", "hello there"])
            >>> tokenizer.vocab
            {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3, 'hello': 4}
        """
        word_counts = Counter()

        # Tokenize and count all words
        for text in texts:
            tokens = self._tokenize(text)
            word_counts.update(tokens)

        # Add words above frequency threshold
        for word, count in word_counts.items():
            if count >= min_freq and word not in self.vocab:
                self.vocab[word] = self.vocab_size
                self.vocab_size += 1

        # Build reverse vocabulary
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def _tokenize(self, text):
        """
        Tokenize a text string into lowercase words, removing punctuation.

        Args:
            text (str): Input text to tokenize.

        Returns:
            list[str]: List of lowercase word tokens.

        Example:
            >>> tokenizer = Tokenizer()
            >>> tokenizer._tokenize("Hello, world!")
            ['hello', 'world']
        """
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()

    def encode(self, text, max_length=None, add_special_tokens=False):
        """
        Convert a text string into a sequence of token IDs.

        Args:
            text (str): Input text to encode.
            max_length (int, optional): Desired fixed length. If specified:
                - The sequence will be truncated if longer than `max_length`.
                - The sequence will be padded with `<PAD>` if shorter.
            add_special_tokens (bool, optional): If True, adds `<START>` at the
                beginning and `<END>` at the end. Defaults to False.

        Returns:
            np.ndarray: Numpy array of token IDs (dtype=int).

        Example:
            >>> tokenizer = Tokenizer()
            >>> tokenizer.fit(["hello world"])
            >>> tokenizer.encode("hello world", max_length=5)
            array([4, 1, 0, 0, 0])
        """
        tokens = self._tokenize(text)

        if add_special_tokens:
            tokens = ["<START>"] + tokens + ["<END>"]

        encoded = [self.vocab.get(token, self.special_tokens["<UNK>"]) for token in tokens]

        if max_length is not None:
            if len(encoded) > max_length:
                encoded = encoded[:max_length]
            else:
                encoded.extend([self.special_tokens["<PAD>"]] * (max_length - len(encoded)))

        return np.array(encoded, dtype=np.int32)

    def decode(self, encoded, skip_special_tokens=True):
        """
        Convert a sequence of token IDs back into a text string.

        Args:
            encoded (list[int] | np.ndarray): List or array of token IDs.
            skip_special_tokens (bool, optional): If True, removes special tokens from the output.
                Defaults to True.

        Returns:
            str: Decoded text string.

        Example:
            >>> tokenizer.decode([2, 4, 5, 3])
            'hello world'
        """
        tokens = [self.reverse_vocab.get(int(idx), "<UNK>") for idx in encoded]

        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.special_tokens]

        return " ".join(tokens)
