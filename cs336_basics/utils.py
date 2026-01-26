import regex as re


# Provided in the assignment description.
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def pre_tokenize_chunk(chunk: str, special_pattern_str: str | None) -> dict[tuple[bytes], int]:
    """Regex tokenizes the chunk. Splits first on special tokens, then uses PAT."""
    freqs: dict[tuple[bytes], int] = {}
    special_pattern = re.compile(special_pattern_str) if special_pattern_str else None
    sub_chunks = special_pattern.split(chunk) if special_pattern else [chunk]

    for sub_chunk in sub_chunks:
        for match in PAT.finditer(sub_chunk):
            match_bytes = tuple(bytes([b]) for b in match.group().encode("UTF-8"))
            freqs[match_bytes] = freqs.get(match_bytes, 0) + 1

    return freqs
