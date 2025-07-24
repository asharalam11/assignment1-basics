from argparse import ArgumentParser

from cs336_basics.bpe.train import train_bpe_tokenizer
from cs336_basics.bpe.utils import save_vocab_and_merges

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
        default=500,
        help="Size of the vocabulary.",
    )
    parser.add_argument(
        "--special_tokens",
        type=str,
        nargs="+",
        default=["<pad>", "<unk>", "<s>", "</s>"],
        help="List of special tokens to be used.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/out"),
        help="Output directory",
    )
    

    args = parser.parse_args()

    vocab, merges = train_bpe_tokenizer(args.input_path, args.vocab_size, args.special_tokens)
    save_vocab_and_merges(vocab=vocab, merges=merges, output_dir=args.output_dir)

if __name__ == "__main__":
    main()