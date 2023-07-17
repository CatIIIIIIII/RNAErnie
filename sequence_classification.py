"""
This module implements sequence classification.

Author: wangning(wangning.roci@gmail.com)
Date  : 2022/12/7 7:42 PM
"""
# built-in modules
import math
import os.path as osp
import time
from collections import defaultdict
# third-party modules
from tqdm import tqdm
import numpy as np
from Bio import SeqIO
# paddle modules
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset
from paddlenlp.transformers import ErniePretrainedModel
# self-defined modules
from dataset_utils import seq2input_ids
from base_classes import (
    BaseTrainer,
    BaseCollator,
    BaseInstance,
    BaseFuncMetrics,
)


class SeqClsDataset(Dataset):
    """.fasta Dataset for sequence classification.
    """

    def __init__(self, fasta_dir, prefix, tokenizer, seed=0, train=True, num_specials=2):
        """init function for SeqClsDataset

        Args:
            fasta_dir (str): fasta directory
            prefix (str): classificaiton task name
            tokenizer (tokenizer_nuc.NUCTokenizer): nucleotide tokenizer
            seed (int, optional): random seed. Defaults to 0.
            train (bool, optional): whether train dataset. Defaults to True.
            num_specials (int, optional): number of special tokens in sentence. Defaults to 2.
        """
        super(SeqClsDataset, self).__init__()

        self.fasta_dir = fasta_dir
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.num_specials = num_specials

        file_name = "train.fa" if train else "test.fa"
        fasta = osp.join(osp.join(fasta_dir, prefix), file_name)
        records = list(SeqIO.parse(fasta, "fasta"))
        self.data = [(str(x.seq), x.description.split(" ")[1]) for x in records]
        np_rng = np.random.RandomState(seed=seed)
        np_rng.shuffle(self.data)

    def __getitem__(self, idx):
        """get item from dataset

        Args:
            idx (int): index of data

        Returns:
            dict: item of dataset
        """
        instance = self.data[idx]
        seq = instance[0]
        label = instance[1]
        return {"seq": seq, "label": label}

    def __len__(self):
        """get length of dataset
        Returns:
            int: length of dataset
        """
        return len(self.data)


class ClsInstance(BaseInstance):
    """A single instance for sequence classification.
    """

    def __init__(self, input_ids, label):
        """
        Args:
            input_ids: for ernie model (either from pretrained or scratch)
            label:

        Returns:
            None
        """
        super(ClsInstance, self).__init__()
        self.input_ids = input_ids
        self.label = label
        self.length = len(input_ids)


class SeqClsCollator(BaseCollator):
    """Data collator for sequence classification.
    """

    def __init__(self, max_seq_len, tokenizer):
        """
        Args:
            max_seq_len: max sequence length for input
            tokenizer:

        Returns:
            None
        """
        super(SeqClsCollator, self).__init__()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def __call__(self, data):
        """
        Args:
            data: instance for stack

        Returns:
            dict for trainer
        """
        sep_id = self.tokenizer.vocab.token_to_idx[self.tokenizer.sep_token]

        input_ids_stack = []
        labels_stack = []
        seq_lens_stack = []

        if not self.max_seq_len:
            self.max_seq_len = max([x.length for x in data])

        for i_batch in range(len(data)):
            instance = data[i_batch]

            input_ids = getattr(instance, "input_ids")
            if len(input_ids) > self.max_seq_len:
                # move [SEP] to beginning
                input_ids = input_ids[:self.max_seq_len]
                input_ids[-1] = sep_id
            input_ids += [0] * (self.max_seq_len - len(input_ids))
            input_ids_stack.append(input_ids)

            labels_stack.append(getattr(instance, "label"))
            seq_lens_stack.append(getattr(instance, "length"))

        return {
            "input_ids": self.stack_fn(input_ids_stack),
            "labels": self.stack_fn(labels_stack),
            "seq_lens": self.stack_fn(seq_lens_stack)
        }


class LongSeqClsCollator(BaseCollator):
    """Data collator for sequence classification.
    """

    def __init__(self, max_seq_len, tokenizer):
        """
        Args:
            max_seq_len: max sequence length for input
            tokenizer:

        Returns:
            None
        """
        super(LongSeqClsCollator, self).__init__()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.chunk_size = 512

    def __call__(self, data):
        """
        Args:
            data: instance for stack

        Returns:
            dict for trainer
        """
        cls_id = self.tokenizer.vocab.token_to_idx[self.tokenizer.cls_token]
        sep_id = self.tokenizer.vocab.token_to_idx[self.tokenizer.sep_token]

        input_ids_stack = []
        labels_stack = []
        seq_lens_stack = []

        # set max length
        self.max_seq_len = min(max([x.length for x in data]), self.max_seq_len)
        num_chunk = math.ceil(self.max_seq_len / self.chunk_size)
        self.max_seq_len = num_chunk * self.chunk_size

        for i_batch in range(len(data)):
            instance = data[i_batch]

            input_ids = getattr(instance, "input_ids")
            if len(input_ids) > self.max_seq_len:
                input_ids = input_ids[:self.max_seq_len]
            input_ids += [0] * (self.max_seq_len - len(input_ids))

            # chunk whole sequence by chunk_size
            for i in range(1, num_chunk):
                input_ids.insert(i * self.chunk_size - 1, sep_id)
                input_ids.insert(i * self.chunk_size, cls_id)
            input_ids.insert(num_chunk * self.chunk_size - 1, sep_id)
            input_ids = input_ids[:self.max_seq_len]
            input_ids_chunk = [input_ids[i:i + self.chunk_size] for i in range(0, self.max_seq_len, self.chunk_size)]
            chunk_lens = [len(i) for i in input_ids_chunk]
            input_ids_chunk = self.stack_fn(input_ids_chunk)
            input_ids_stack.append(input_ids_chunk)

            labels_stack.append(getattr(instance, "label"))
            seq_lens_stack.append(chunk_lens)

        return {
            "input_ids": self.stack_fn(input_ids_stack),  # [B, Ch, L]
            "labels": self.stack_fn(labels_stack),  # [B]
            "seq_lens": self.stack_fn(seq_lens_stack)
        }


def convert_instance_to_cls(raw_data, tokenizer, label2id):
    """
    Args:
        raw_data:
        tokenizer:
        label2id: map str label to int

    Returns:
        ClsInstance
    """
    seq = raw_data["seq"]
    label = raw_data["label"]

    seq = seq.upper().replace("U", "T")
    input_ids = seq2input_ids(seq, tokenizer)

    label_id = label2id[label]

    return ClsInstance(input_ids=input_ids, label=label_id)


class SeqClsLoss(nn.Layer):
    """Loss of sequence classification.
    """

    def __init__(self):
        """
        Returns:
            None
        """
        super(SeqClsLoss, self).__init__()

    def forward(self, outputs, labels, topk_probs=None):
        """forward function
        Args:
            outputs: [B, C] logit scores
            labels: [N] labels
            topk_probs: [B, top_k] probabilities of top k indications

        Returns:
            Tensor: final loss.
        """
        labels = paddle.cast(labels, dtype='int64')
        if topk_probs is not None:
            assert isinstance(topk_probs, paddle.Tensor)
            assert outputs.shape[0] == np.prod(topk_probs.shape)

            top_k = topk_probs.shape[-1]

            labels = paddle.repeat_interleave(labels, top_k)

            loss = F.cross_entropy(outputs, labels, reduction='none')
            loss = paddle.reshape(loss, shape=(-1, top_k))
            loss = paddle.multiply(loss, topk_probs)
            loss = paddle.sum(loss, axis=1)
            loss = paddle.mean(loss)
        else:
            loss = F.cross_entropy(outputs, labels)
        return loss


class SeqClsMetrics(BaseFuncMetrics):
    """Metrics for classification
    """

    def __call__(self, outputs, labels):
        """
        Args:
            outputs: output of model, (batch_size, )
            labels: ground truth of data, (batch_size, )

        Returns:
            metrics in dict
        """
        preds = paddle.argmax(outputs, axis=-1)
        preds = paddle.cast(preds, 'int32')
        preds = preds.numpy()

        labels = paddle.cast(labels, 'int32')
        labels = labels.numpy()

        res = {}
        for name in self.metrics:
            func = getattr(self, name)
            if func:
                m = func(preds, labels)
                res[name] = m
            else:
                raise NotImplementedError
        return res


class ErnieForLongSequenceClassification(ErniePretrainedModel):
    """
    Ernie Model with a linear layer on top of the output layer,
    designed for sequence classification/regression tasks like GLUE tasks.

    Args:
        ernie (ErnieModel): An instance of `paddlenlp.transformers.ErnieModel`.
        num_classes (int, optional): The number of classes. Default to `2`.
    """

    def __init__(self, ernie, num_classes=2):
        super(ErnieForLongSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.ernie = ernie  # allow ernie to be config
        self.dropout = nn.Dropout(self.ernie.config["hidden_dropout_prob"])
        self.classifier = nn.Linear(self.ernie.config["hidden_size"], num_classes)
        self.apply(self.init_weights)

    def forward(self, input_ids):
        """
        Args:
            input_ids (Tensor): (B, Ch, L)

        Returns:
            logits
        """
        (B, Ch, L) = input_ids.shape
        # (B*Ch, L)
        input_ids = paddle.reshape(input_ids, shape=(-1, L))
        outputs = self.ernie(input_ids)
        pooled_output = outputs[1]  # (B*Ch, hidden_size)

        pooled_output = self.dropout(pooled_output)
        # (B*CH, num_classes)
        logits = self.classifier(pooled_output)
        # (B, Ch, num_classes)
        logits_Ch = paddle.reshape(logits, shape=(B, Ch, -1))
        # (B, num_classes)
        logits_avg = paddle.mean(logits_Ch, axis=1)
        return logits_avg


class SeqClsTrainer(BaseTrainer):
    """Trainer for sequence classification.
    """

    def __init__(self,
                 args,
                 tokenizer,
                 model,
                 train_dataset,
                 eval_dataset=None,
                 data_collator=None,
                 loss_fn=None,
                 optimizer=None,
                 compute_metrics=None,
                 visual_writer=None):
        """
        Args:
            args:
            model:
            train_dataset:
            eval_dataset:
            data_collator:
            loss_fn:
            optimizer:
            compute_metrics:
            visual_writer:

        Returns:
            None
        """
        super(SeqClsTrainer, self).__init__(
            args=args,
            tokenizer=tokenizer,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            loss_fn=loss_fn,
            optimizer=optimizer,
            compute_metrics=compute_metrics,
            visual_writer=visual_writer
        )

    def train(self, epoch):
        """
        Returns:
            None
        """
        self.model.train()
        time_st = time.time()
        with tqdm(total=len(self.train_dataset), disable=self.args.disable_tqdm) as pbar:
            for i, data in enumerate(self.train_dataloader):
                input_ids = data["input_ids"]
                labels = data["labels"]

                outputs = self.model(input_ids)
                loss = self.loss_fn(outputs, labels)

                self.optimizer.clear_grad()
                loss.backward()
                self.optimizer.step()

                # log to pbar
                pbar.set_postfix(train_loss='{:.4f}'.format(loss.item()))
                pbar.update(self.args.batch_size)
                # log to visualdl
                if (i + 1) % self.args.logging_steps == 0:
                    # log to directory
                    tag_value = {"train/loss": loss.item()}
                    self.visual_writer.update_scalars(tag_value=tag_value, step=self.args.logging_steps)

        time_ed = time.time() - time_st
        print('Train\tLoss: {:.6f}; Time: {:.4f}'.format(loss.item(), time_ed))

    def eval(self, epoch):
        """
        Returns:
            None
        """
        self.model.eval()
        time_st = time.time()
        with tqdm(total=len(self.eval_dataset), disable=self.args.disable_tqdm) as pbar:
            outputs_dataset, labels_dataset = [], []
            for i, data in enumerate(self.eval_dataloader):
                input_ids = data["input_ids"]
                labels = data["labels"]

                with paddle.no_grad():
                    outputs = self.model(input_ids)

                metrics = self.compute_metrics(outputs, labels)

                pbar.set_postfix(accuracy='{:.4f}'.format(metrics[self.name_pbar]))
                pbar.update(self.args.logging_steps)

                outputs_dataset.append(outputs)
                labels_dataset.append(labels)

        # save best model
        outputs_dataset = paddle.concat(outputs_dataset, axis=0)
        labels_dataset = paddle.concat(labels_dataset, axis=0)
        metrics_dataset = self.compute_metrics(outputs_dataset, labels_dataset)
        if self.args.save_max:
            self.save_model(metrics_dataset, epoch)

        # log results to screen/bash
        results = {}
        log = 'Test\t' + self.args.dataset + "\t"
        # log results to visualdl
        tag_value = defaultdict(float)
        # extract results
        for k, v in metrics_dataset.items():
            log += k + ": {" + k + ":.4f}\t"
            results[k] = v
            tag = "eval/" + k
            tag_value[tag] = v
        self.visual_writer.update_scalars(tag_value=tag_value, step=1)

        time_ed = time.time() - time_st
        print('Test\tTime: {:.4f}'.format(time_ed))
        print(log.format(**results))
