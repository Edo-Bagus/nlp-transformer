"""
GUI for Decoder-Only Transformer (GPT-style)
--------------------------------------------
Complies with assignment requirements:
1. Uses only NumPy for mathematical operations.
2. No deep learning libraries (PyTorch, TensorFlow, etc.).
3. Modular structure (Transformer, Tokenizer, GUI separated).
4. Input = simple integer tokenization.
5. Output = logits [batch, seq_len, vocab_size] and softmax probabilities
   for next-token prediction.

Author: <Your Name>
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from transformer import Transformer
from tokenizer import Tokenizer


class TransformerGUI:
    """
    Graphical interface for visualizing Transformer next-token prediction.
    Uses NumPy-only Transformer and simple tokenizer.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("NumPy Transformer — Next Token Predictor")

        # === Model Parameters ===
        self.num_layers = 2
        self.d_model = 128
        self.num_heads = 8
        self.d_ff = 512
        self.max_seq_length = 50
        self.vocab_size = 1000

        # === Tokenizer Setup ===
        self.tokenizer = Tokenizer()
        self._train_tokenizer()

        np.random.seed(10)

        # === Initialize Transformer ===
        self.model = Transformer(
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            vocab_size=self.vocab_size,
            max_seq_length=self.max_seq_length,
        )

        # === Build GUI Layout ===
        self._create_widgets()

    # -------------------------------
    # Tokenizer Setup
    # -------------------------------
    def _train_tokenizer(self):
        """Train tokenizer on small sample corpus."""
        sample_texts = [
            "hello world how are you",
            "what is your name",
            "nice to meet you",
            "machine learning is fun",
            "artificial intelligence project",
            "natural language processing",
            "python programming language",
            "deep learning models",
            "transformer architecture",
            "data science and ai"
        ]
        self.tokenizer.fit(sample_texts)
        self.vocab_size = self.tokenizer.vocab_size
        print(f"✅ Tokenizer trained — vocabulary size: {self.vocab_size}")

    # -------------------------------
    # GUI Layout
    # -------------------------------
    def _create_widgets(self):
        """Creates all GUI elements."""
        # Input Frame
        input_frame = ttk.LabelFrame(self.root, text="Input Text", padding="10")
        input_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")

        self.input_text = tk.Text(input_frame, height=4, width=60)
        self.input_text.grid(row=0, column=0, padx=5, pady=5)

        predict_btn = ttk.Button(input_frame, text="Predict Next Token", command=self.predict_next_token)
        predict_btn.grid(row=1, column=0, pady=5)

        # Output Frame
        output_frame = ttk.LabelFrame(self.root, text="Model Output", padding="10")
        output_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        self.tree = ttk.Treeview(output_frame, columns=("Token", "Probability"), show="headings")
        self.tree.heading("Token", text="Predicted Token")
        self.tree.heading("Probability", text="Softmax Probability")
        self.tree.column("Token", width=200)
        self.tree.column("Probability", width=150)
        self.tree.grid(row=0, column=0, padx=5, pady=5)

        # Grid configuration
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

    # -------------------------------
    # Prediction Function
    # -------------------------------
    def predict_next_token(self):
        """
        Tokenizes the input text, runs forward pass through the NumPy-only
        Transformer, and displays top-5 next-token probabilities.
        """
        # 1️⃣ Read input
        input_text = self.input_text.get("1.0", tk.END).strip()
        if not input_text:
            return

        # 2️⃣ Tokenize input → integer sequence
        encoded_input = self.tokenizer.encode(input_text, max_length=self.max_seq_length)
        encoded_input = np.array([encoded_input], dtype=np.int32)

        # 3️⃣ Forward pass (no training, just inference)
        logits, probs = self.model.forward(encoded_input)

        # 4️⃣ Clear old predictions
        for item in self.tree.get_children():
            self.tree.delete(item)

        # 5️⃣ Get top-5 predictions
        top_indices = np.argsort(probs[0])[::-1][:5]
        for idx in top_indices:
            token = self.tokenizer.reverse_vocab.get(idx, "<UNK>")
            prob = probs[0][idx]
            self.tree.insert("", "end", values=(token, f"{prob:.4f}"))

        # 6️⃣ (Optional) Debugging Info
        print("----- Model Debug Info -----")
        print(f"Input shape: {encoded_input.shape}")
        print(f"Logits shape: {logits.shape}")
        print(f"Probs shape: {probs.shape}")
        print(f"Sum of probs: {np.sum(probs[0]):.4f}")
        print("-----------------------------")


# --------------------------------------
# Run the GUI Application
# --------------------------------------
def main():
    root = tk.Tk()
    app = TransformerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
