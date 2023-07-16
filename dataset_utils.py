"""
This module defines useful sequence process funtions.

Author: wangning(wangning.roci@gmail.com)
Date  : 2023/3/13 15:06
"""


def seq2kmer(seq, k_mer, max_length=None):
    """
    convert sequence to kmer split.
    Args:
        seq:
        k_mer:
        max_length:

    Returns:
        kmer sequence
    """
    kmer_text = ""
    i = 0
    upper = len(seq) - k_mer + 1
    if max_length:
        upper = min(upper, max_length)
    while i < upper:
        kmer_text += (seq[i: i + k_mer] + " ")
        i += 1
    kmer_text = kmer_text.strip()
    return kmer_text


def seq2input_ids(seq, tokenizer):
    """
    convert sequence to input ids
    Args:
        seq:
        tokenizer:

    Returns:
        tokenizer input ids
    """
    k_mer = tokenizer.k_mer

    seq += seq[-1] * (k_mer - 1)
    kmer_text = ""
    i = 0
    while i < len(seq) - k_mer + 1:
        kmer_text += (seq[i: i + k_mer] + " ")
        i += 1
    kmer_text = kmer_text.strip()
    tokens = tokenizer(text=kmer_text,
                       return_token_type_ids=False)
    input_ids = tokens["input_ids"]
    return input_ids


def batch_converter(data, tokenizer, max_seq_len=None):
    """
    Args:
        data:
        tokenizer:
        max_seq_len:

    Returns:
        batch input ids
    """
    if max_seq_len is None:
        max_seq_len = max([len(x) for x in data])

    batch_input_ids = []
    for seq in data:
        seq = seq.upper().replace("U", "T")
        input_ids = seq2input_ids(seq, tokenizer)
        input_ids += [0] * (max_seq_len - len(input_ids))
        batch_input_ids.append(input_ids)

    return batch_input_ids
