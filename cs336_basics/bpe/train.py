import regex as re

from pathlib import Path
from tqdm import tqdm
from .utils import read_text_file
from .pretokenization_example import find_chunk_boundaries

# Pretokenization Regex pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""



"""
The above PAT leads to pretokenization like this
Pretoken:  (b'\n', b'\n', b'\n')
Pretoken:  (b'\n', b'\n', b'\n')
Pretoken:  (b'\n', b'\n')
Pretoken:  (b' ', b'\n', b' ')
Pretoken:  (b' ', b'\n', b' ')
Pretoken:  (b'\n', b'\n', b' ')
Pretoken:  (b'\n', b'\n', b' ')
Pretoken:  (b' ', b'\n')
Pretoken:  (b' ', b'\n', b'\n')
Pretoken:  (b' ', b'\n', b'\n')
Pretoken:  (b' ', b'\n', b'\n', b'\n')
Pretoken:  (b' ', b'\n', b'\n', b'\n')
Pretoken:  (b' ', b'\n', b'\n', b'\n')
Pretoken:  (b'\n', b' ')
Pretoken:  (b' ', b'\n', b'\n', b' ')
Pretoken:  (b' ', b'\n', b'\n', b' ')
Pretoken:  (b' ', b'\n', b'\n', b' ')

which creates an error in the pytest
uv run pytest tests/test_train_bpe.py::test_train_bpe_special_tokens

we get an extra merge due to pretokenization: {(b'\n\n', b'\n')}

"""


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
    assert vocab_size >= (256 + len(special_tokens)), (
        f"Vocabulary size must be greater than {256 + len(special_tokens)}, but got {vocab_size}."
    )

    # Create a mapping from IDs to byte strings
    vocab = {i: bytes([i]) for i in range(256)}
    
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

    return "".join(split_chunk)

def run_pretokenization(input_path: Path, special_tokens: list[str], num_chunks: int) -> dict[bytes, int]:
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
        i = 0
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            i += 1
            f.seek(start)

            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            
            # Remove special tokens
            chunk = remove_special_tokens(chunk_str=chunk, special_tokens=special_tokens)

            # Run pre-tokenization on your chunk and store the counts for each pre-token
            chunk_iterator = re.finditer(pattern=PAT, string=chunk)

            for match in chunk_iterator:
                word = match.group(0)
                
                encoded = word.encode("utf-8")
                word_as_bytes = tuple(bytes([b]) for b in encoded)
                pretoken_counts[word_as_bytes] = pretoken_counts.get(word_as_bytes, 0) + 1

            # TODO: Remove once ready to test the whole program
            print("Processing chunk #: ", i)
            
    return pretoken_counts

def bpe_merge(pretoken_counts: dict[bytes, int], vocab: dict[int, bytes], vocab_size: int) -> tuple[dict[int, bytes], list[bytes]]:
    """
    Function to perform bpe_merges

    Args:
        pretoken_counts (dict[bytes, int]): Pretoken counts
        vocab (dict[int, bytes]): Vocabulary 
        vocab_size (int): Maximum vocabulary size

    Returns:
        tuple[dict[int, bytes], list[bytes]]: Return the final vocab and merges
    """

    # Construct pair frequency from the pretoken counts
    pair_counts = {}
    for pretoken, count in pretoken_counts.items():
        for i, j in zip(pretoken, pretoken[1:]):
            # Debug unit test: tests/test_train_bpe.py::test_train_bpe_special_tokens
            # if i == b'\n' or j == b'\n':
            #     print("Pretoken: ", pretoken)
            pair_counts[(i, j)] = pair_counts.get((i, j), 0) + count
    
    # print("Pair frequency table: ", pair_counts)

    merges = []
    
    # Continue performing merges until 
    num_merges = vocab_size - len(vocab)
    for _ in tqdm(range(num_merges)):
        # Find the most frequent pair
        most_freq = max(pair_counts, key= lambda k: (pair_counts.get(k), k))

        merges.append(most_freq)

        new_token = b"".join(most_freq)
        vocab.update({len(vocab): new_token})

        # Perform merges (two loops - 1. Over the pretoken counts 2. Over each pretoken -> update both pretoken_counts and pair_counts)
        updated_pretoken_count = {}
        for pretoken, count in pretoken_counts.items():
            skip_idx= False
            new_pretoken = ()
            for idx, (i, j) in enumerate(zip(pretoken, pretoken[1:])):
                # I really wanted to use zip hence had to use this
                # Easier index manipulation could have been done with a while loop
                if skip_idx:
                    skip_idx=False
                    continue
                
                pair = b"".join((i, j))
                if new_token == pair:
                    if new_pretoken:
                        # Extract the new indices for prefix and suffixfrom this pretoken with one less as 
                        # idx has moved cz of the merge even after the skip but only for the prefix, suffix
                        prefix = new_pretoken[:idx-1]
                        suffix = new_pretoken[idx + 1:]
                    else:
                        prefix = pretoken[:idx]
                        suffix = pretoken[idx + 2:]
                    new_pretoken = prefix + (new_token,) + suffix
                    if prefix:
                        old_left_pair = (prefix[-1], i)
                        pair_counts[old_left_pair] -= count
                        new_left_pair = (prefix[-1], pair)
                        pair_counts.update({new_left_pair: pair_counts.get(new_left_pair, 0) + count})
                    if suffix:
                        old_right_pair = (j, suffix[0])
                        pair_counts[old_right_pair] -= count
                        new_right_pair = (pair, suffix[0])
                        pair_counts.update({new_right_pair: pair_counts.get(new_right_pair, 0)+count})

                    pair_counts[most_freq] -= count
                    skip_idx = True

            if new_pretoken:
                updated_pretoken_count[new_pretoken] = count
            else:
                updated_pretoken_count[pretoken] = count

        pretoken_counts = updated_pretoken_count

    return vocab, merges
    

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
    pretoken_counts = run_pretokenization(input_path=input_path, special_tokens=special_tokens, num_chunks=1)
    # print(pretoken_counts)
    # Step 3: BPE merge loop
    vocab, merges = bpe_merge(pretoken_counts=pretoken_counts, vocab=vocab, vocab_size=vocab_size)
    # print(vocab)
    # print(merges)
    # Step 4: Assign token IDs

    # Step 5: Return the vocabulary and merges


    file_content = read_text_file(input_path)

    # print("Length of file content:", len(file_content))
    # print("First 100 bytes of file content:", file_content[:100])
    # print("Special tokens:", special_tokens)
    return vocab, merges
