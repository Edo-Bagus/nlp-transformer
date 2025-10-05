import numpy as np


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


def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Applies Layer Normalization over the last dimension of the input.

    Args:
        x (np.ndarray): Input tensor of shape (..., d_model).
        eps (float): Small constant for numerical stability.

    Returns:
        np.ndarray: Layer-normalized tensor with same shape as input.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + eps)


def scaled_dot_product_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes scaled dot-product attention.

    Args:
        Q (np.ndarray): Query tensor of shape (batch_size, num_heads, seq_len_q, depth).
        K (np.ndarray): Key tensor of shape (batch_size, num_heads, seq_len_k, depth).
        V (np.ndarray): Value tensor of shape (batch_size, num_heads, seq_len_k, depth).
        mask (np.ndarray | None): Optional mask broadcastable to (batch_size, num_heads, seq_len_q, seq_len_k).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - output: (batch_size, num_heads, seq_len_q, depth)
            - attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
    """
    d_k = K.shape[-1]

    # Transpose last two dimensions of K for matrix multiplication
    K_t = np.swapaxes(K, -1, -2)
    attention_scores = np.matmul(Q, K_t) / np.sqrt(d_k)

    if mask is not None:
        # Apply large negative value to masked positions
        attention_scores += (mask * -1e9)

    # Compute attention weights and final weighted output
    attention_weights = softmax(attention_scores, axis=-1)
    output = np.matmul(attention_weights, V)

    return output, attention_weights


class MultiHeadAttention:
    """
    Implements Multi-Head Attention mechanism as described in the Transformer architecture.

    Attributes:
        d_model (int): Dimensionality of model embeddings.
        num_heads (int): Number of attention heads.
        depth (int): Dimension per head = d_model / num_heads.
        wq, wk, wv, wo (np.ndarray): Learnable projection matrices.
    """

    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        # Initialize linear projection weights (small random values)
        self.wq = np.random.randn(d_model, d_model) * 0.1
        self.wk = np.random.randn(d_model, d_model) * 0.1
        self.wv = np.random.randn(d_model, d_model) * 0.1
        self.wo = np.random.randn(d_model, d_model) * 0.1

    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Splits the last dimension into (num_heads, depth) and transposes for attention computation.

        Args:
            x (np.ndarray): Tensor of shape (batch_size, seq_len, d_model).

        Returns:
            np.ndarray: Tensor of shape (batch_size, num_heads, seq_len, depth).
        """
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.depth)
        return x.transpose(0, 2, 1, 3)

    def forward(
        self, q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Forward pass through the Multi-Head Attention layer.

        Args:
            q, k, v (np.ndarray): Input tensors of shape (batch_size, seq_len, d_model).
            mask (np.ndarray | None): Optional attention mask.

        Returns:
            np.ndarray: Output tensor of shape (batch_size, seq_len, d_model).
        """
        batch_size = q.shape[0]

        # Linear projections
        q = np.dot(q, self.wq)
        k = np.dot(k, self.wk)
        v = np.dot(v, self.wv)

        # Split into multiple heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Scaled dot-product attention
        scaled_attention, _ = scaled_dot_product_attention(q, k, v, mask)

        # Concatenate heads back
        scaled_attention = scaled_attention.transpose(0, 2, 1, 3)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)

        # Final linear projection
        output = np.dot(concat_attention, self.wo)

        return output


class PositionWiseFFN:
    """
    Implements the position-wise feed-forward network used in Transformer blocks.

    Architecture:
        FFN(x) = max(0, xW1)W2  (ReLU activation)

    Attributes:
        w1 (np.ndarray): First linear layer weights (d_model, d_ff).
        w2 (np.ndarray): Second linear layer weights (d_ff, d_model).
    """

    def __init__(self, d_model: int, d_ff: int):
        self.w1 = np.random.randn(d_model, d_ff) * 0.1
        self.w2 = np.random.randn(d_ff, d_model) * 0.1

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the feed-forward network.

        Args:
            x (np.ndarray): Input tensor (batch_size, seq_len, d_model).

        Returns:
            np.ndarray: Output tensor (batch_size, seq_len, d_model).
        """
        hidden = np.dot(x, self.w1)
        activated = np.maximum(0, hidden)  # ReLU
        return np.dot(activated, self.w2)


def positional_encoding(length: int, d_model: int) -> np.ndarray:
    """
    Generates sinusoidal positional encodings for sequences.

    Args:
        length (int): Maximum sequence length.
        d_model (int): Embedding dimension.

    Returns:
        np.ndarray: Positional encoding of shape (1, length, d_model).
    """
    positions = np.arange(length)[:, np.newaxis]
    div_terms = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    angle_rads = positions * div_terms

    # Apply sin to even indices, cos to odd indices
    pos_encoding = np.zeros((length, d_model))
    pos_encoding[:, 0::2] = np.sin(angle_rads)
    pos_encoding[:, 1::2] = np.cos(angle_rads)

    return pos_encoding[np.newaxis, ...]  # (1, length, d_model)
