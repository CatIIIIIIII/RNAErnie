"""
This module inference sequences embeddings.

Author: wangning(wangning.roci@gmail.com)
Date  : 2022/12/7 7:41 PM
"""

from Bio import SeqIO

import paddle
from paddlenlp.utils.log import logger
from paddlenlp.data import Stack
from paddlenlp.transformers import ErnieModel

from dataset_utils import seq2input_ids
from tokenizer_nuc import NUCTokenizer


class BatchConverter(object):
    """Convert sequences to batch inputs.
    """

    def __init__(self, k_mer=1, vocab_path="./data/vocab/vocab_1MER.txt", batch_size=256, max_seq_len=512, is_pad=True, st_pos=0):
        """this class predicts embeddings from RNA sequences list

        Args:
            k_mer (int): k for splitting RNA sequences
            vocab_path (str): root path to vocab file
            batch_size (int): batch size
            max_seq_len (int): max sequence to input sequences
            st_pos (int): start position of the sequence
        """
        self.tokenizer = NUCTokenizer(
            k_mer=k_mer,
            vocab_file=vocab_path,
        )
        self.stack_fn = Stack()

        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.is_pad = is_pad
        self.st_pos = st_pos

    def __call__(self, data):
        """call function to convert sequences to batch inputs.

        Args:
            data (List[tuple(name, sequence)], str): data or .fasta file to be converted
        """

        if type(data) == list:
            # convert sequences to batch inputs
            self.data = data
        elif type(data) == str and (data.split(".")[-1] == "fasta" or data.split(".")[-1] == "fa"):
            # load .fasta file with SeqIO
            records = list(SeqIO.parse(data, "fasta"))
            self.data = [(str(x.description), str(x.seq)) for x in records]
        # return generator to iterate data by step batch_size
        for d in range(0, len(self.data), self.batch_size):
            raw_data = self.data[d:d + self.batch_size]
            names = [x[0] for x in raw_data]
            seqs = [x[1] for x in raw_data]
            seqs = [x[self.st_pos:self.st_pos + self.max_seq_len - 2] for x in seqs]
            seqs = [x.upper().replace("U", "T") for x in seqs]
            # 2 means [CLS] and [SEP]
            input_ids = [seq2input_ids(x, self.tokenizer) for x in seqs]
            if self.is_pad:
                input_ids = [x + [0] * (self.max_seq_len - len(x)) for x in input_ids]
                input_ids = self.stack_fn(input_ids)
            input_ids = paddle.to_tensor(input_ids)
            yield names, seqs, input_ids


if __name__ == "__main__":
    # ========== Set device
    logger.debug("Set device.")
    paddle.set_device("gpu")

    # ========== Prepare Data
    data = [
        ("RNA1", "GGGUGCGAUCAUACCAGCACUAAUGCCCUCCUGGGAAGUCCUCGUGUUGCACCCCU"),
        ("RNA2", "GGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCA"),
        ("RNA3", "CGAUUCNCGUUCCC--CCGCCUCCA"),
    ]
    data = "./data/ft/seq_cls/nRC/test.fa"

    # ========== Batch Converter
    logger.debug("Loading converter.")
    batch_converter = BatchConverter(k_mer=1,
                                     vocab_path="./data/vocab/vocab_1MER.txt",
                                     batch_size=256,
                                     max_seq_len=512)

    # ========== RNAErnie Model
    rna_ernie = ErnieModel.from_pretrained("./output/BERT,ERNIE,MOTIF,PROMPT/checkpoint-final")

    # call batch_converter to convert sequences to batch inputs
    for names, _, inputs_ids in batch_converter(data):
        with paddle.no_grad():
            # extract whole sequence embeddings
            embeddings = rna_ernie(inputs_ids)[0].detach()
            # extract [CLS] token embedding
            embeddings = embeddings[:, 0, :]
            print(embeddings.shape)
