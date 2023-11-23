
def seq2kmer(seq, k_mer, max_length=None):
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
