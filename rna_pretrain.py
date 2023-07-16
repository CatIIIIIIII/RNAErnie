"""
This module builds rna pretraining.

Author: wangning(wangning.roci@gmail.com)
Date  : 2022/9/8 1:21 PM
"""

# built-in modules
import random
from paddlenlp.data import Stack
import numpy as np
import os.path as osp
import collections
from dataclasses import dataclass, field
# 3rd-party modules
from Bio import SeqIO
# paddle modules
import paddle
from paddle.io import DistributedBatchSampler, Dataset
from paddlenlp.trainer import Trainer
from paddlenlp.trainer import TrainingArguments
# self-defined modules
from dataset_utils import seq2kmer


def add_start_docstrings(*docstr):
    """add docstring to function
    """
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class PreTrainingArguments(TrainingArguments):
    """args for pretraining training

    Args:
        TrainingArguments (paddlenlp.trainer.TrainingArguments): args for training
    """
    decay_steps: int = field(
        default=None,
        metadata={"help": "Decay ending point in learning rate scheduler."}
    )
    min_learning_rate: float = field(
        default=None,
        metadata={"help": "Minimum learning rate deacyed to."},
    )
    output_dir: str = field(
        default="./",
        metadata={"help": "Output directory."}
    )


class PreFastaDataset(Dataset):
    def __init__(self, fasta_dir, prefix, num_file, num_samples_per_file, tokenizer, max_model_length=512, num_specials=4, replace=True):
        """init pretrain fasta dataset

        Args:
            fasta_dir (str): fasta file directory
            prefix (int): prefix of fasta file
            num_file (int): number of fasta file
            num_samples_per_file (int): number of samples in each fasta file
            tokenizer (tokenizer_nuc.NUCTokenizer): nucleotide tokenizer
            max_model_length (int, optional): maximum input sequence length. Defaults to 512.
            num_specials (int, optional): number of special tokens in sequence. Defaults to 4.
            replace (bool, optional): whether replace 'U' with 'T'. Defaults to True.
        """
        super().__init__()

        self.fasta_dir = fasta_dir
        self.prefix = prefix
        self.num_samples = num_file * num_samples_per_file
        self.num_samples_per_file = num_samples_per_file
        self.tokenizer = tokenizer
        self.max_model_length = max_model_length
        # 4 special tokens are: [CLS], [IND], [LABEL], [SEP]
        self.num_specials = num_specials
        self.replace = replace
        self.cache = {}

    def __getitem__(self, idx):
        """get item from dataset

        Args:
            idx (int): index of dataset

        Returns:
            dict: {kmer_text: str}
        """
        # assert idx>=0, "Training number must large than checkpoints have seen."
        idx_file = idx // self.num_samples_per_file
        idx_sample = idx % self.num_samples_per_file
        if idx_file not in self.cache:
            file_name = self.prefix + "_" + str(idx_file + 1) + ".fasta"
            fasta = osp.join(self.fasta_dir, file_name)
            records = list(SeqIO.parse(fasta, "fasta"))
            data = [(str(x.seq), x.description.split(" ")[1]) for x in records]
            # labels = [ for x in records]
            self.cache = {}
            self.cache[idx_file] = data

        (seq, label) = self.cache[idx_file][idx_sample]
        label = label.replace("_", "")
        if self.replace:
            seq = seq.replace("U", "T")

        # cut into vocabs by k_mer
        kmer_text = seq2kmer(seq=seq, k_mer=self.tokenizer.k_mer, max_length=self.max_model_length - self.num_specials)
        kmer_text += (" " + label)

        return {"kmer_text": kmer_text}

    def __len__(self):
        """get dataset length

        Returns:
            int: dataset length
        """
        return self.num_samples


def load_motif(motif_dir, motif_name, tokenizer):
    """load motifs from file

    Args:
        motif_dir (str): motif data root directory
        motif_name (str): motif file name
        tokenizer (tokenizer_nuc.NUCTokenizer): nucleotide tokenizer

    Returns:
        dict: {file_name: [int]}
    """
    res = {}
    for name in motif_name.split(","):
        motif_path = osp.join(motif_dir, name + ".txt")
        with open(motif_path, 'r') as f:
            motifs = f.readlines()

        motif_tokens = []
        for m in motifs:
            kmer_text = seq2kmer(seq=m, k_mer=1)
            input_ids = tokenizer(kmer_text, return_token_type_ids=False)["input_ids"]
            input_ids = input_ids[1:-1]
            motif_tokens.append(input_ids)
        res[name] = motif_tokens

    return res


class PreTrainingInstance(object):
    """A single training instance.
    """

    def __init__(self, input_ids, input_mask, masked_lm_positions, masked_lm_labels):
        """init PreTrainingInstance

        Args:
            input_ids (list): input ids of RNA sequence
            input_mask (list): input mask of RNA sequence
            masked_lm_positions (list): masking positions
            masked_lm_labels (list): masking labels
        """
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels
        self.length = len(input_ids)

    def __call__(self):
        """call function

        Returns:
            dict: {self.input_ids, self.input_mask, self.masked_lm_positions, self.masked_lm_labels}
        """
        return vars(self).items()


class PreDataCollator:
    """Data collator that will dynamically pad the inputs to the longest sequence in the batch and process them to model.
    """

    def __init__(self, pad_token_id, stack_fn=Stack()):
        """init PreDataCollator

        Args:
            pad_token_id (int): padding token id
            stack_fn (paddlenlp.data.Stack, optional): stacking function. Defaults to Stack().
        """
        self.pad = pad_token_id
        self.stack_fn = stack_fn

    def __call__(self, data):
        """call function

        Args:
            data (list): PreTrainingInstance list

        Returns:
            dict: stacked data
        """
        # Padding
        max_length = max([x.length for x in data])
        # print(max_length)
        for x in data:
            remains = max_length - x.length

            x.input_ids += [self.pad] * remains
            x.input_mask += [0] * remains
        # Stacking
        num_fields = len(data[0]())
        out = [None] * num_fields
        stack_names = {
            0: "input_ids",
            1: "input_mask",
        }
        for i, name in stack_names.items():
            out[i] = self.stack_fn([getattr(x, name) for x in data])
        # Padding for divisibility by 8 for fp16 or int8 usage
        size = sum(len(getattr(x, "masked_lm_positions")) for x in data)
        if size % 8 != 0:
            size += 8 - (size % 8)
        # masked_lm_positions
        # Organize as a 1D tensor for gather or use gather_nd
        out[2] = np.full(size, 0, dtype=np.int32)
        # masked_lm_labels
        out[3] = np.full([size, 1], -1, dtype=np.int64)
        mask_token_num = 0
        batch_size, seq_length = out[0].shape
        for i, x in enumerate(data):
            for j, pos in enumerate(getattr(x, "masked_lm_positions")):
                out[2][mask_token_num] = i * seq_length + pos
                out[3][mask_token_num] = getattr(x, "masked_lm_labels")[j]
                mask_token_num += 1
        return {
            "input_ids": out[0],
            "attention_mask": out[1],
            "masked_positions": out[2],
            "labels": out[3],
        }


def convert_text_to_pretrain(raw_data, tokenizer, pre_strategy, max_seq_length, masked_lm_prob, motif_tree_dict, seed):
    """convert raw data to pretrain data

    Args:
        raw_data (dict): raw data
        tokenizer (tokenizer_nuc.NUCTokenizer): nucleotide tokenizer
        pre_strategy (str): pretrain strategy, split by ","
        max_seq_length (int): max sequence length
        masked_lm_prob (float): language model masking probability
        motif_tree_dict (dict): motif tree dict
        seed (int): random seed

    Returns:
        PreTrainingInstance: pretrain instance
    """
    tokenized_texts = tokenizer(text=raw_data['kmer_text'],
                                return_token_type_ids=False)
    # special tokens
    tokens = tokenized_texts['input_ids']
    IND = tokenizer.vocab.token_to_idx[tokenizer.ind_token]
    CLS = tokens[0]
    LABEL = tokens[-2]
    SEP = tokens[-1]
    # remove "[CLS]", "LABEL" and "[SEP]"
    tokens = tokens[1:-2]
    # masking
    max_predictions_num = masked_lm_prob * max_seq_length
    np_rng = np.random.RandomState(seed=(seed + tokenized_texts['input_ids'][1]) % 2 ** 32)

    stages = []
    if "BERT" in pre_strategy:
        input_ids, masked_positions, masked_labels = create_masked_lm_predictions_bert(
            tokens,
            tokenizer,
            masked_lm_prob,
            max_predictions_num,
            np_rng)
        stages += [(input_ids, masked_positions, masked_labels, "BERT")]
    if "ERNIE" in pre_strategy:
        input_ids, masked_positions, masked_labels = create_masked_lm_predictions_ernie(
            tokens,
            tokenizer,
            masked_lm_prob,
            max_predictions_num,
            np_rng)
        stages += [(input_ids, masked_positions, masked_labels, "ERNIE")]
    if "MOTIF" in pre_strategy:
        input_ids, masked_positions, masked_labels = create_masked_lm_predictions_motif(
            tokens,
            tokenizer,
            masked_lm_prob,
            max_predictions_num,
            motif_tree_dict,
            np_rng)
        stages += [(input_ids, masked_positions, masked_labels, "MOTIF")]

    (input_ids, masked_positions, masked_labels, stage_name) = random.choice(stages)
    if "PROMPT" in pre_strategy:
        input_ids = [CLS] + input_ids + [IND] + [LABEL] + [SEP]
    else:
        input_ids = [CLS] + input_ids + [SEP]
    assert len(input_ids) <= 512, f"input length ({len(input_ids)}) exceed model max length"

    masked_positions = [x + 1 for x in masked_positions]  # add [CLS] at beginning
    input_mask = [1] * len(input_ids)
    res = PreTrainingInstance(input_ids=input_ids,
                              input_mask=input_mask,
                              masked_lm_positions=masked_positions,
                              masked_lm_labels=masked_labels)
    return res


def create_masked_lm_predictions_bert(tokens, tokenizer, masked_lm_prob, max_predictions_per_seq, rng):
    """create masked language model predictions by BERT strategy

    Args:
        tokens (list): input tokens id
        tokenizer (tokenizer_nuc.NUCTokenizer): nucleotide tokenizer
        masked_lm_prob (float): language model masking probability
        max_predictions_per_seq (int): number of max predictions per sequence
        rng (np.random.RandomState): random state

    Returns:
        tuple: input ids, masked positions, masked labels
    """

    MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                              ["index", "label"])

    mask = tokenizer.mask_token_id
    normal_vocab_words = tokenizer.normal_vocab_id_list

    cand_indexes = list(range(len(tokens)))
    rng.shuffle(cand_indexes)

    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
    output_tokens = list(tokens)
    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        # 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
            masked_token = mask
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = normal_vocab_words[rng.randint(0, len(normal_vocab_words))]

        output_tokens[index] = masked_token
        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return output_tokens, masked_lm_positions, masked_lm_labels


def create_masked_lm_predictions_ernie(tokens, tokenizer, masked_lm_prob, max_predictions_per_seq, rng, max_ngrams=3):
    """create masked language model predictions by ERNIE strategy

    Args:
        tokens (list): input tokens id
        tokenizer (tokenizer_nuc.NUCTokenizer): nucleotide tokenizer
        masked_lm_prob (float): language model masking probability
        max_predictions_per_seq (int): number of max predictions per sequence
        rng (np.random.RandomState): random state
        max_ngrams (int): extended ngrams for masking range

    Returns:
        tuple: input ids, masked positions, masked labels
    """
    MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                              ["index", "label"])

    mask = tokenizer.mask_token_id
    normal_vocab_words = tokenizer.normal_vocab_id_list

    k_mer = tokenizer.k_mer
    if k_mer == 1:
        # this means that mask phrases range in (5, 8) when we mask 1_mer tokens
        # todo: unify 1_mer with other k_mers
        k_mer = 5
        max_ngrams = 4
    cand_indexes = list(range(len(tokens) - (k_mer + max_ngrams)))

    output_tokens = list(tokens)

    masked_lm_positions = []
    masked_lm_labels = []

    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))

    ngrams = np.arange(k_mer, k_mer + max_ngrams, dtype=np.int64)
    pvals = 1. / np.arange(1, max_ngrams + 1)
    pvals /= pvals.sum(keepdims=True)

    ngram_indexes = []
    for idx in range(len(cand_indexes)):
        ngram_index = []
        for n in ngrams:
            ngram_index.append(cand_indexes[idx:idx + n])
        ngram_indexes.append(ngram_index)

    rng.shuffle(ngram_indexes)

    masked_lms = []
    covered_indexes = set()
    backup_output_tokens = list(output_tokens)
    for cand_index_set in ngram_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if not cand_index_set:
            continue
        # Skip current piece if they are covered in lm masking or previous ngrams.
        for index_set in cand_index_set:
            for index in index_set:
                if index in covered_indexes:
                    continue

        i = rng.choice(
            np.arange(len(cand_index_set)),
            p=pvals,
        )

        # Note(mingdachen):
        # Repeatedly looking for a candidate that does not exceed the
        # maximum number of predictions by trying shorter ngrams.
        index_set = cand_index_set[i]
        while len(masked_lms) + len(index_set) > num_to_predict:
            i -= 1
            if i < 0:
                break
            index_set = cand_index_set[i]
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = mask
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = output_tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = normal_vocab_words[rng.randint(0, len(normal_vocab_words))]
            output_tokens[index] = masked_token
            # plus 1 because we remove [CLS] at first
            masked_lms.append(MaskedLmInstance(index=index, label=backup_output_tokens[index]))

    assert len(masked_lms) <= num_to_predict

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return output_tokens, masked_lm_positions, masked_lm_labels


def create_masked_lm_predictions_motif(
        tokens,
        tokenizer,
        masked_lm_prob,
        max_predictions_per_seq,
        motif_trees,
        np_rng):
    """create masked language model predictions by motif strategy

    Args:
        tokens (list): input tokens id
        tokenizer (tokenizer_nuc.NUCTokenizer): nucleotide tokenizer
        masked_lm_prob (float): language model masking probability
        max_predictions_per_seq (int): number of max predictions per sequence
        motif_trees (dict): motif tree dict
        rng (np.random.RandomState): random state

    Returns:
        tuple: input ids, masked positions, masked labels
    """

    MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                              ["index", "label"])

    mask = tokenizer.mask_token_id
    normal_vocab_words = tokenizer.normal_vocab_id_list

    output_tokens = list(tokens)
    masked_lm_positions = []
    masked_lm_labels = []
    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))

    ngram_indexes = []
    # motif_name = random.sample(motif_trees.keys(), 1)
    motif_tree = motif_trees["DataBases"]
    search_results = motif_tree.search_all(tokens)
    for result in search_results:
        ngram_index = list(range(result[1], result[1] + len(result[0])))
        ngram_indexes.append([ngram_index])

    np_rng.shuffle(ngram_indexes)

    # mask reminding tokens with statistics if former not full
    motif_tree = motif_trees["Statistics"]
    search_results = motif_tree.search_all(tokens)
    for result in search_results:
        ngram_index = list(range(result[1], result[1] + len(result[0])))
        ngram_indexes.append([ngram_index])

    masked_lms = []
    covered_indexes = set()
    backup_output_tokens = list(output_tokens)
    for cand_index_set in ngram_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if not cand_index_set:
            continue
        # Skip current piece if they are covered in lm masking or previous ngrams.
        for index_set in cand_index_set:
            for index in index_set:
                if index in covered_indexes:
                    continue

        index_set = cand_index_set[0]
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            # 80% of the time, replace with [MASK]
            if np_rng.random() < 0.8:
                masked_token = mask
            else:
                # 10% of the time, keep original
                if np_rng.random() < 0.5:
                    masked_token = output_tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = normal_vocab_words[np_rng.randint(0, len(normal_vocab_words))]
            output_tokens[index] = masked_token
            # plus 1 because we remove [CLS] at first
            masked_lms.append(MaskedLmInstance(index=index, label=backup_output_tokens[index]))

    assert len(masked_lms) <= num_to_predict

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return output_tokens, masked_lm_positions, masked_lm_labels


class CriterionWrapper(paddle.nn.Layer):
    """wrapper for criterion
    """

    def __init__(self, criterion):
        """CriterionWrapper
        """
        super(CriterionWrapper, self).__init__()
        self.criterion = criterion

    def forward(self, output, labels):
        """forward function
        Args:
            output (tuple): prediction_scores, seq_relationship_score
            labels (tuple): masked_lm_labels, next_sentence_labels
        Returns:
            Tensor: final loss.
        """
        masked_lm_labels = labels
        prediction_scores = output
        loss = self.criterion(prediction_scores, None, masked_lm_labels)
        return loss


class PretrainingTrainer(Trainer):
    """paddle trainer for pretraining

    Args:
        Trainer (paddlenlp.trainer.Trainer): model pretraining trainer
    """

    def __init__(self, *args, **kwargs):
        """init trainer
        """
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self):
        """get pretraining sampler

        Returns:
            BatchSampler (paddle.io.DistributedBatchSampler): pretraining sampler
        """
        if not isinstance(self.train_dataset, collections.abc.Sized):
            return None

        if self.args.world_size <= 1:
            return paddle.io.BatchSampler(
                dataset=self.train_dataset,
                shuffle=False,
                batch_size=self.args.per_device_train_batch_size,
                drop_last=self.args.dataloader_drop_last)

        return DistributedBatchSampler(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            drop_last=self.args.dataloader_drop_last
        )
