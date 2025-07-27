import json
import regex as re

from collections.abc import Iterable, Iterator
from tqdm import tqdm

from .utils import bytes_to_unicode
from .train import PAT

class Tokenizer:
    """
    Tokenizer class to encode and decode text based on learnt vocab and merges
    """
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]=None):
        
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.inverse_vocab = {value: key for key, value in vocab.items()}
        # reorganzie merges into pair -> new token id dict
        self.merge_dict = {}
        for a, b in merges:
            id_pair = (self.inverse_vocab[a], self.inverse_vocab[b])
            self.merge_dict[id_pair] = self.inverse_vocab[a+b]

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str]=None):
        """
        Class method to build a Tokenizer from a serialized vocabulary and list of merges.
        """
        # Copied from tests/test_tokenizer.py
        byte2unicode = bytes_to_unicode()

        gpt2_byte_decoder = {v: k for k, v in byte2unicode.items()}
        with open(vocab_filepath) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
        # just return the original bytes, so we don't force students to use
        # any particular encoding scheme.
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        
        return cls(vocab, merges, special_tokens)

    def encode_chunk(self, chunk: str) -> list[int]:
        """
        Encode a chunk of text into a sequence of token ids:
        """
        token_ids = []

        # Pretokenize 
        chunk_iterator = re.finditer(pattern=PAT, string=chunk)

        for match in chunk_iterator:
            word = match.group(0)
            encoded = word.encode("utf-8")

            ids = [self.inverse_vocab[bytes([b])] for b in encoded]

            while len(ids) >= 2:
                pairs = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
                
                # The lower index merge is more important as that merge happened first, so is the highest frequency merge
                most_important_pair = min(pairs, key=lambda pair: self.merge_dict.get(pair, float('inf')))

                if most_important_pair not in self.merge_dict:
                    break
                
                # Else perform the merge
                new_ids = [] 
                i = 0
                while i < len(ids):
                    curr_pair = tuple(ids[i:i+2])
                    # Perform the merge
                    if curr_pair == most_important_pair:
                        new_ids.append(self.merge_dict[most_important_pair])
                        i += 1
                    else:
                        new_ids.append(ids[i])
                    i += 1
                ids = new_ids
            token_ids.extend(ids)

        return token_ids

    def encode(self, text: str) -> list[int]:
        """
        Encode and input text into a sequence of token IDs.
        """

        # Split on special tokens
        if self.special_tokens:   
            chunked_text = re.split("|".join(re.escape(token) for token in self.special_tokens), text)
        else:
            chunked_text = [text]

        token_ids = []
        for chunk in tqdm(chunked_text, desc=f"Encoding {len(chunked_text)} documents"):
            token_ids.extend(self.encode_chunk(chunk))

        return token_ids


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Return a generator that yields token IDs given an iterable of strings
        """
        for text in iterable:
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id

    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text
        """
        text = []
        for id in ids:
            if id in self.vocab:
                text.append(self.vocab[id])
        
        text_as_bytes = b''.join(text)
        return text_as_bytes.decode("utf-8", errors="replace")
