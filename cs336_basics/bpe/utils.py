import os
import json
from functools import lru_cache
from pathlib import Path

# https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def read_text_file(file_path: Path) -> str:
    """
    Read a text file and return its content as a string.
    
    Args:
        file_path (Path): Path to the text file.
        
    Returns:
        str: Content of the file.
    """
    with open(file_path, 'rb') as file:
        return file.read()
    
def save_vocab_and_merges(vocab:tuple[dict[int, bytes]], merges:list[tuple[bytes, bytes]], output_dir: Path):
    """
    Function to save the vocab and merges returne after training bpe

    Args: 
        dict[int, bytes]: Mapping from token IDs to byte strings.
        list[tuple[bytes, bytes]]: List of pairs of byte strings representing the BPE merges.
    """

    # Create output dir if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = Path(output_dir, "tiny_stories_vocab.json")
    merges_path = Path(output_dir, "tiny_stories_merges.txt")

    byte2unicode = bytes_to_unicode()

    # Convert byte tokens in vocab to printable strings
    string_vocab = {
        ''.join(byte2unicode[b] for b in token): idx
        for idx, token in vocab.items()
    }

    # Convert byte-pair merges to printable string format
    string_merges = [
        ' '.join(
            ''.join(byte2unicode[b] for b in part)
            for part in merge
        )
        for merge in merges
    ]

    with open(vocab_path, "w", encoding="utf-8") as file:
        json.dump(string_vocab, file, indent=4)

    # Save the merges list to a file
    with open(merges_path, 'w', encoding='utf-8') as f:
        for merge in string_merges:
            f.write(merge + '\n')