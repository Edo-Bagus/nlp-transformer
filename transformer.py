import numpy as np
from layers import MultiHeadAttention, PositionWiseFFN, layer_norm, positional_encoding


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Computes the softmax of an input array along the specified axis.

    Args:
        x (np.ndarray): Input array.
        axis (int): Axis along which to compute softmax.

    Returns:
        np.ndarray: Softmax probabilities with same shape as `x`.
    """
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Creates a causal mask to prevent attention to future tokens.

    The mask is an upper-triangular matrix filled with -1e9 (a large negative value)
    above the main diagonal, used to mask future positions in attention computation.

    Args:
        seq_len (int): Length of the sequence.

    Returns:
        np.ndarray: Causal mask of shape (1, 1, seq_len, seq_len).
    """
    mask = np.triu(np.ones((1, 1, seq_len, seq_len)), k=1)
    return mask * -1e9


class DecoderLayer:
    """
    Represents a single decoder layer in a decoder-only Transformer (GPT-style).

    Each layer performs:
        1. LayerNorm → Multi-Head Self-Attention → Residual Add
        2. LayerNorm → Position-wise Feed Forward Network → Residual Add

    Attributes:
        mha (MultiHeadAttention): Multi-head self-attention module.
        ffn (PositionWiseFFN): Feed-forward network.
        dropout_rate (float): Dropout probability (not used in NumPy implementation).
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        """
        Initializes the decoder layer.

        Args:
            d_model (int): Model embedding dimensionality.
            num_heads (int): Number of attention heads.
            d_ff (int): Dimensionality of feed-forward network hidden layer.
            dropout_rate (float): Dropout rate (not applied in this NumPy version).
        """
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFFN(d_model, d_ff)
        self.dropout_rate = dropout_rate

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """
        Performs a forward pass through the decoder layer.

        Args:
            x (np.ndarray): Input tensor of shape (batch_size, seq_len, d_model).
            mask (np.ndarray | None): Optional causal mask for self-attention.

        Returns:
            np.ndarray: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # === Pre-Norm before attention ===
        norm_x = layer_norm(x)
        attn_output = self.mha.forward(norm_x, norm_x, norm_x, mask)
        x = x + attn_output  # Residual connection

        # === Pre-Norm before FFN ===
        norm_x = layer_norm(x)
        ffn_output = self.ffn.forward(norm_x)
        x = x + ffn_output  # Residual connection

        return x


class Transformer:
    """
    Implements a simplified GPT-style (decoder-only) Transformer using NumPy.

    The model includes:
        - Token and positional embeddings
        - A stack of decoder layers
        - Final linear projection to vocabulary logits
        - Softmax over the final position for next-token prediction

    Attributes:
        num_layers (int): Number of decoder layers.
        d_model (int): Dimensionality of token embeddings.
        embedding (np.ndarray): Token embedding matrix (vocab_size, d_model).
        pos_encoding (np.ndarray): Positional encoding matrix (1, max_seq_len, d_model).
        layers (list[DecoderLayer]): Stack of decoder layers.
        final_linear (np.ndarray): Final projection weight (d_model, vocab_size).
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        max_seq_length: int,
        dropout_rate: float = 0.1,
    ):
        """
        Initializes the Transformer model.

        Args:
            num_layers (int): Number of decoder layers.
            d_model (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            d_ff (int): Hidden layer size in feed-forward network.
            vocab_size (int): Vocabulary size.
            max_seq_length (int): Maximum sequence length supported.
            dropout_rate (float): Dropout rate (not applied in NumPy version).
        """
        self.num_layers = num_layers
        self.d_model = d_model

        # Embedding matrices
        self.embedding = np.random.randn(vocab_size, d_model) * 0.1
        self.pos_encoding = positional_encoding(max_seq_length, d_model)

        # Stack of decoder layers
        self.layers = [
            DecoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)
        ]

        # Output projection layer
        self.final_linear = np.random.randn(d_model, vocab_size) * 0.1

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs a forward pass through the Transformer.

        This includes embedding lookup, adding positional encodings,
        passing through decoder layers, projecting to vocabulary logits,
        and computing softmax over the last token.

        Args:
            x (np.ndarray): Input token IDs of shape (batch_size, seq_len).

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - logits: Raw output scores before softmax, shape (batch_size, seq_len, vocab_size).
                - probs: Softmax probabilities for the last token, shape (batch_size, vocab_size).
        """
        batch_size, seq_len = x.shape

        # Token + positional embeddings
        x = self.embedding[x]  # (batch_size, seq_len, d_model)
        x += self.pos_encoding[:, :seq_len, :]

        # Causal mask to block future positions
        mask = create_causal_mask(seq_len)

        # Pass through each decoder layer
        for layer in self.layers:
            x = layer.forward(x, mask)

        # Project to vocabulary space
        logits = np.dot(x, self.final_linear)  # (batch_size, seq_len, vocab_size)

        # Compute softmax only for the last token (next-token prediction)
        probs = softmax(logits[:, -1, :])

        return logits, probs
