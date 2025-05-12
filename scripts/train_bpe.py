from argparse import ArgumentParser

from cs336_basics.bpe.train import train_bpe_tokenizer
from pathlib import Path

def main():
    parser = ArgumentParser(description="Train a BPE tokenizer.")
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="Path to the input text file.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=10000,
        help="Size of the vocabulary.",
    )
    parser.add_argument(
        "--special_tokens",
        type=str,
        nargs="+",
        default=["<pad>", "<unk>", "<s>", "</s>"],
        help="List of special tokens to be used.",
    )

    args = parser.parse_args()

    train_bpe_tokenizer(args.input_path, args.vocab_size, args.special_tokens)

if __name__ == "__main__":
    main()