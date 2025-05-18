import regex as re

from pathlib import Path
from .utils import read_text_file
from .pretokenization_example import find_chunk_boundaries

# Pretokenization Regex pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def initialize_vocabulary(
    special_tokens: list[str], vocab_size: int
) -> dict[int, bytes]:
    """
    Initialize the vocabulary with special tokens.

    Args:
        special_tokens (list[str]): List of special tokens to be used.
        vocab_size (int): Size of the vocabulary.

    Returns:
        dict[int, bytes]: Mapping from token IDs to byte strings.
    """
    assert vocab_size > (256 + len(special_tokens)), (
        f"Vocabulary size must be greater than (256 + len(special_tokens)), but got {vocab_size}."
    )

    # Create a mapping from IDs to byte strings
    vocab = {i: chr(i).encode("utf-8") for i in range(256)}

    # Create a mapping from token IDs to byte strings
    vocab.update({i + 256: token.encode("utf-8") for i, token in enumerate(special_tokens)})
    
    return vocab

def remove_special_tokens(chunk_str: str, special_tokens: list[str]) -> str:
    """
    Remove special tokens before pre-tokenization

    Args:
        chunk_str (str): Chunked section of string
        special_tokens (list[str]): List of special tokens

    Returns:
        str: Sanitized string
    """
    # Split on special tokens
    split_chunk = re.split("|".join(re.escape(token) for token in special_tokens), chunk_str)
    
    return " ".join(split_chunk)

def run_pretokenization(input_path: Path, special_tokens: list[str], num_chunks: int) -> dict[int, bytes]:
    """
    Chunk the file at boundaries and run pretokenization

    Args:  
        input_path (Path): Input file path
        special_tokens (list[str]): List of special tokens
        num_chunks (int): Number of chunks to be made

    Returns:
        dict[str, int]: Pretokenization counts
    """

    pretoken_counts = {}

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")
        
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)

            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            
            # Remove special tokens
            chunk = remove_special_tokens(chunk_str=chunk, special_tokens=special_tokens)

            # Run pre-tokenization on your chunk and store the counts for each pre-token
            chunk_iterator = re.finditer(pattern=PAT, string=chunk)
            while True:
                try:
                    word = next(chunk_iterator).group(0)
                    pretoken_counts.update({word: pretoken_counts.get(word, 0) + 1})
                except StopIteration:
                    break
            # TODO: Remove once ready to test the whole program
            break
        
    return pretoken_counts



def train_bpe_tokenizer(
    input_path: Path, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the given input file.

    Args:
        input_path (Path): Path to the input text file.
        vocab_size (int): Size of the vocabulary.
        special_tokens (list[str]): List of special tokens to be used.

    Returns:
        dict[int, bytes]: Mapping from token IDs to byte strings.
        list[tuple[bytes, bytes]]: List of pairs of byte strings representing the BPE merges.
    """
    
    # Step 1: Inititalize the vocabulary
    vocab = initialize_vocabulary(special_tokens, vocab_size)

    # Step 2: Read the input file and pretokenize it
    pretoken_counts = run_pretokenization(input_path=input_path, special_tokens=special_tokens, num_chunks=50)
    print(pretoken_counts)
    # Step 3: BPE merge loop

    # Step 4: Assign token IDs

    # Step 5: Return the vocabulary and merges


    file_content = read_text_file(input_path)

    print("Length of file content:", len(file_content))
    print("First 100 bytes of file content:", file_content[:100])
    print("Special tokens:", special_tokens)
    return vocab
