# GPT-Style Transformer from Scratch (NumPy)

This is a simple implementation of a decoder-only Transformer (GPT-style) using **NumPy**.  
It includes a word-level tokenizer, multi-head attention, feed-forward layers, and positional encoding.

## Dependencies

- Python 3.x
- NumPy

Install all depedencies if you haven't:

```bash
pip install -r requiremts.txt
```

## Running the Program

1. Clone the repository
2. Make sure all files (`main.py`, `layers.py`, `tokenizers.py`, `transformers.py`) are in the same folder.
3. Run the main script:

```bash
python main.py
```

The program will execute the forward pass of the Transformer and output predictions or relevant logs, depending on the code in main.py.
