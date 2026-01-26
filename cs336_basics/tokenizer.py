from collections.abc import Iterable, Iterator
import regex as re
import pickle

from cs336_basics import utils


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        """
        Constructs a tokenizer from a vocab, list of merges, and (optionally) list of special tokens.
        """
        # token id to token bytes
        self.vocab = vocab
        # token bytes to token id
        self.vocab_inv = {v: k for k, v in vocab.items()}
        self.merges = merges
        # pair to index rank
        self.merges_dict = {merge: i for i, merge in enumerate(merges)}
        # cache mapping pretokens (strings) to their encoded token IDs (list of ints)
        self.encode_cache = {}
        self.cache_hits = 0

        self.pretokenize_pattern = re.compile(utils.PAT)

        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            self.special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"

            next_id = max(self.vocab.keys()) + 1
            for token in special_tokens:
                token_bytes = token.encode("UTF-8")
                if token_bytes not in self.vocab_inv:
                    self.vocab[next_id] = token_bytes
                    self.vocab_inv[token_bytes] = next_id
                    next_id += 1
        else:
            self.special_tokens = None
            self.special_pattern = None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Constructs a Tokenizer from a serialized vocab, list of merges, and (optionally) list of special tokens.
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def decode(self, ids: list[int]) -> str:
        """Decodes a sequence of token IDs into text."""
        text = b"".join(self.vocab[id] for id in ids)
        return text.decode("UTF-8", errors="replace")

    def _pretokenize(self, text: str) -> list[str]:
        """Splits text into 'pretokens' and builds an initial byte representation for each."""
        pretokens: list[str] = []

        for match in self.pretokenize_pattern.finditer(text):
            match_str = match.group()
            pretokens.append(match_str)

        return pretokens

    def encode(self, text: str) -> list[int]:
        """Encodes an input text into a sequence of token IDs, handling special tokens."""
        if not self.special_tokens:
            return self._encode_chunk(text)

        # If we have special tokens, split on them, keeping delimiters
        special_chunks = re.split(self.special_pattern, text)

        ids = []
        for part in special_chunks:
            if part in self.special_tokens:
                # this is a special token
                ids.append(self.vocab_inv[part.encode("UTF-8")])
            else:
                # this is ordinary text
                ids.extend(self._encode_chunk(part))
        return ids

    def _encode_chunk(self, text: str) -> list[int]:
        """Encodes an input text chunk into a sequence of token IDs."""
        pretokens = self._pretokenize(text)
        pretoken_reprs: dict[str, list[bytes]] = {}

        ids = []

        # Merge each pretoken using the BPE rules, in ascending rank order
        for p in pretokens:
            # Check if this pretoken has already been encoded and cached
            if p in self.encode_cache:
                ids.extend(self.encode_cache[p])
                self.cache_hits += 1
            else:
                # Each character → single bytes: e.g. "abc" -> [b'a', b'b', b'c'] match_bytes
                # "a时" => [b'a', b'\xe6', b'\x97', b'\xb6'] match_bytes
                if p not in pretoken_reprs:
                    match_bytes = list(bytes([b]) for b in p.encode("UTF-8"))
                    pretoken_reprs[p] = match_bytes

                merged = self._merge_subword(pretoken_reprs[p])
                token_ids = [self.vocab_inv[subword] for subword in merged]
                # Cache the encoded token IDs for this pretoken
                self.encode_cache[p] = token_ids
                ids.extend(token_ids)

        return ids

    def _merge_subword(self, rep: list[bytes]) -> list[bytes]:
        """
        Given a list of subword units (bytes), repeatedly merges adjacent pairs
        in ascending rank order until no more merges are found.
        
        Example:
            Input: [b'h', b'e', b'l', b'l', b'o']
            Merge dict: {(b'l', b'l'): 2, (b'h', b'e'): 5, (b'e', b'll'): 10, (b'he', b'll'): 15, (b'hell', b'o'): 20}
            
            Step 1: Best pair is (b'l', b'l') with rank 2
                    Merge → [b'h', b'e', b'll', b'o']
            Step 2: Best pair is (b'h', b'e') with rank 5
                    Merge → [b'he', b'll', b'o']
            Step 3: Best pair is (b'e', b'll') with rank 10
                    No match found (we have b'he', not b'e' as separate token)
            Step 4: Best pair is (b'he', b'll') with rank 15
                    Merge → [b'hell', b'o']
            Step 5: Best pair is (b'hell', b'o') with rank 20
                    Merge → [b'hello']
            Step 6: No more pairs to merge → return [b'hello']
        """
        while True:
            best_rank = float("inf")
            best_idx = None

            # Scan adjacent pairs
            for i in range(len(rep) - 1):
                pair = (rep[i], rep[i + 1])
                rank = self.merges_dict.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_idx = i

            # If no merges found, we're done
            if best_idx is None:
                return rep

            # Merge the best pair
            merged = rep[best_idx] + rep[best_idx + 1]  # Concatenate bytes
            rep = rep[:best_idx] + [merged] + rep[best_idx + 2 :]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Yields token IDs lazily from an iterable of strings (e.g., a file handle)."""
        for text in iterable:
            yield from self.encode(text)