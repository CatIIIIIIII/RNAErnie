"""
This module builds rna rna interaction functions.

Author: wangning(wangning.roci@gmail.com)
Date  : 2022/12/7 7:41 PM
"""

import time
import os.path as osp
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import paddle
import paddle.nn as nn
from paddlenlp.data import Stack
from paddle.io import Dataset

from dataset_utils import seq2input_ids
from base_classes import BaseMetrics, BaseTrainer, MlpProjector


class GenerateRRInterTrainTest:
    """generate train and test dataset for rna rna interaction prediction.
    """

    def __init__(self,
                 rr_dir,
                 dataset,
                 split=0.8,
                 seed=0):
        """init function

        Args:
            rr_dir (str): data root dir
            dataset (str): dataset name
            split (float, optional): split ratio. Defaults to 0.8.
            seed (int, optional): random seed. Defaults to 0.
        """

        csv_path = osp.join(osp.join(rr_dir, dataset), dataset + ".csv")
        self.data = pd.read_csv(csv_path, sep=",").values.tolist()

        self.split_index = int(len(self.data) * split)

        np_rng = np.random.RandomState(seed=seed)
        np_rng.shuffle(self.data)

    def get(self):
        """get train and test dataset

        Returns:
            tuple: RRInterDataset, RRInterDataset
        """
        return RRInterDataset(self.data[:self.split_index]), RRInterDataset(self.data[self.split_index:])


class RRInterDataset(Dataset):
    """rna rna interaction dataset
    """

    def __init__(self, data):
        """init function

        Args:
            data (list): rna rna interaction data
        """
        super().__init__()
        self.data = data

    def __getitem__(self, idx):
        """get item

        Args:
            idx (int): index of data

        Returns:
            dict: data
        """
        instance = self.data[idx]
        return {
            "a_name": instance[0],
            "a_seq": instance[1],
            "b_name": instance[2],
            "b_seq": instance[3],
            "label": instance[4],
        }

    def __len__(self):
        """get length of dataset

        Returns:
            int: length of dataset
        """
        return len(self.data)


class RRInstance(object):
    """A single fine tuning instance for classification task.
    """

    def __init__(self, name, tokens, input_ids, label):
        """init function

        Args:
            name (str): name of instance
            tokens (list): simple int tokens of instance
            input_ids (list): input ids of instance
            label (int): 0, 1 for negative and positive
        """
        self.name = name
        self.tokens = tokens
        self.input_ids = input_ids
        self.label = label

    def __call__(self):
        """call function

        Returns:
            dict: data
        """
        return vars(self).items()


class RRDataCollator:
    """Data collator that will dynamically pad the inputs to the longest sequence in the batch and process them to model.
    """

    def __init__(self, max_seq_len, stack_fn=Stack()):
        """init function

        Args:
            max_seq_len (int): maximum sequence length
            stack_fn (paddlenlp.data, optional): stacking function. Defaults to Stack().
        """
        self.stack_fn = stack_fn
        self.max_seq_len = max_seq_len

    def __call__(self, data):
        """call function

        Args:
            data (RRInstance): data instance

        Returns:
            dict: data after padding and stacking
        """
        names = []
        tokens_stack = []
        input_ids_stack = []
        labels_stack = []

        max_seq_len = self.max_seq_len

        for i_batch in range(len(data)):
            instance = data[i_batch]

            names.append(getattr(instance, "name"))

            tokens = getattr(instance, "tokens")
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]
            tokens += [0] * (max_seq_len - len(tokens))
            tokens_stack.append(tokens)

            input_ids = getattr(instance, "input_ids")
            if len(input_ids) > max_seq_len:
                input_ids = input_ids[:max_seq_len]
            input_ids += [0] * (max_seq_len - len(input_ids))
            input_ids_stack.append(input_ids)

            labels_stack.append(getattr(instance, "label"))

        return {
            "names": names,
            "tokens": self.stack_fn(tokens_stack),
            "input_ids": self.stack_fn(input_ids_stack),
            "labels": self.stack_fn(labels_stack),
        }


def convert_instance_to_rr(raw_data, tokenizer, max_seq_lens):
    """convert raw data to rna rna interaction instance

    Args:
        raw_data (dict): raw data
        tokenizer (tokenizer_nuc,NUCTokenizer): nucleic acid tokenizer
        max_seq_lens (int): maximun sequence length

    Returns:
        RRInstance: data instance
    """
    a_name = raw_data["a_name"]
    a_seq = raw_data["a_seq"]
    b_name = raw_data["b_name"]
    b_seq = raw_data["b_seq"]
    label = raw_data["label"]

    _, b_max_seq_length = max_seq_lens[0], max_seq_lens[1]
    # encoder maps N,A,T,C,G to 0,1,2,3,4
    encoder = dict(zip('NATCG', range(5)))
    tokens_a = [encoder[x] for x in a_seq.upper()]
    # if len(token_a) > a_max_seq_length:
    #     token_a = token_a[:a_max_seq_length]
    # elif len(token_a) < a_max_seq_length:
    #     token_a = token_a + [0] * (a_max_seq_length - len(token_a))

    tokens_b = [encoder[x] for x in b_seq.upper()]
    if len(tokens_b) > b_max_seq_length:
        tokens_b = tokens_b[:b_max_seq_length]
    elif len(tokens_b) < b_max_seq_length:
        tokens_b = tokens_b + [0] * (b_max_seq_length - len(tokens_b))

    # tokenizer maps N,A,T,C,G to id in vocab file
    a_input_ids = seq2input_ids(a_seq, tokenizer)
    a_input_ids = a_input_ids[1:-1]
    # if len(a_input_ids) > a_max_seq_length:
    #     a_input_ids = a_input_ids[:a_max_seq_length]
    # elif len(a_input_ids) < a_max_seq_length:
    #     a_input_ids = a_input_ids + [0] * (a_max_seq_length - len(a_input_ids))

    b_input_ids = seq2input_ids(b_seq, tokenizer)
    b_input_ids = b_input_ids[1:-1]
    if len(b_input_ids) > b_max_seq_length:
        b_input_ids = b_input_ids[:b_max_seq_length]
    elif len(b_input_ids) < b_max_seq_length:
        b_input_ids = b_input_ids + [0] * (b_max_seq_length - len(b_input_ids))

    name = a_name + "+" + b_name
    tokens = tokens_a + tokens_b
    input_ids = a_input_ids + b_input_ids
    label = [label]

    return RRInstance(name, tokens, input_ids, label)


class RRInterMetrics(BaseMetrics):
    """rna rna interaction metrics
    """

    def __call__(self, outputs, labels):
        """call function
        """

        return super().__call__(outputs, labels)


class ErnieRRInter(nn.Layer):
    """rna rna interaction model.
    """

    NUM_FILTERS = 320
    KERNEL_SIZE = 12
    PROB_DROPOUT = 0.1
    IN_CH = 5

    def __init__(self, extractor,
                 embedding_dim=64,
                 hidden_states_size=0,
                 proj_size=0,
                 with_pretrain=True,
                 fix_pretrain=True):
        """init function

        Args:
            extractor (paddlenlp.transformers.ErnieModel): extract features from pretrained model
            embedding_dim (int, optional): embedding dimension. Defaults to 64.
            hidden_states_size (int, optional): hidden state dimension. Defaults to 0.
            proj_size (int, optional): projectino size. Defaults to 0.
            with_pretrain (bool, optional): whether to pretrain. Defaults to True.
            fix_pretrain (bool, optional): whether to fix pretrained model. Defaults to True.
        """
        super(ErnieRRInter, self).__init__()
        assert (extractor is None) != with_pretrain, "Fail to load pretrained model!"

        n_in = embedding_dim
        if with_pretrain:
            self.extractor = extractor
            self.proj_head = MlpProjector(hidden_states_size, proj_size)
            n_in += proj_size
        self.with_pretrain = with_pretrain
        self.fix_pretrain = fix_pretrain

        self.embedder = nn.Embedding(num_embeddings=5, embedding_dim=embedding_dim)
        self.cnn_lstm = nn.Sequential(
            # num_embeddings=5 means "ATCGN"
            nn.Conv1D(in_channels=n_in, out_channels=self.NUM_FILTERS, kernel_size=self.KERNEL_SIZE),
            nn.ReLU(),
            nn.Dropout(self.PROB_DROPOUT),
            nn.MaxPool1D(kernel_size=2),
            nn.Dropout(self.PROB_DROPOUT),
            nn.LSTM(input_size=27, hidden_size=32, direction="bidirectional"),
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.PROB_DROPOUT),
            nn.Linear(in_features=64, out_features=16),
            nn.Dropout(self.PROB_DROPOUT),
            nn.Linear(in_features=16, out_features=2),
        )

    def forward(self, tokens, input_ids):
        """forward function

        Args:
            tokens (list): simple int tokens of instance
            input_ids (list): input ids of instance

        Returns:
            tuple: logits, last_layer_attn
        """
        x = self.embedder(tokens)
        x = paddle.transpose(x, perm=[0, 2, 1])

        if self.with_pretrain:
            if self.fix_pretrain:
                with paddle.no_grad():
                    outputs = self.extractor(input_ids, output_attentions=True, return_dict=True)
                    embeddings = outputs["last_hidden_state"].detach()
                    last_layer_attn = outputs["attentions"][-1].detach()
            else:
                outputs = self.extractor(input_ids, output_attentions=True, return_dict=True)
                embeddings = outputs["last_hidden_state"].detach()
                last_layer_attn = outputs["attentions"][-1].detach()

            embeddings = self.proj_head(embeddings)
            embeddings = paddle.transpose(embeddings, perm=[0, 2, 1])
            x = paddle.concat([x, embeddings], axis=1)

        x, _ = self.cnn_lstm(x)
        x = x[:, -1, :]
        x = self.classifier(x)

        return x, last_layer_attn


class RRInterCriterionWrapper(paddle.nn.Layer):
    """wrap criterion
    """

    def __init__(self):
        """CriterionWrapper
        """
        super(RRInterCriterionWrapper, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, output, labels):
        """forward function

        Args:
            output (tuple): prediction_scores, seq_relationship_score
            labels (tuple): masked_lm_labels, next_sentence_labels

        Returns:
            Tensor: final loss.
        """
        labels = paddle.cast(labels, dtype='int64')
        loss = self.loss_fn(output, labels)

        return loss


class RRInterTrainer(BaseTrainer):
    """rna rna interaction trainer
    """

    def train(self, epoch):
        """train function

        Args:
            epoch (int): current epoch
        """
        self.model.train()
        time_st = time.time()
        num_total, loss_total = 0, 0

        with tqdm(total=len(self.train_dataset), disable=False) as pbar:
            for i, data in enumerate(self.train_dataloader):
                # names = data["names"]
                tokens = data["tokens"]
                input_ids = data["input_ids"]
                labels = data["labels"]

                preds, _ = self.model(tokens, input_ids)
                loss = self.loss_fn(preds, labels)

                self.optimizer.clear_grad()
                loss.backward()
                self.optimizer.step()

                # log to pbar
                num_total += self.args.batch_size
                loss_total += loss.item()
                pbar.set_postfix(train_loss='{:.4f}'.format(loss_total / num_total))
                pbar.update(self.args.batch_size)
                # reset loss if too many steps
                if num_total >= self.args.logging_steps:
                    num_total, loss_total = 0, 0
                # log to visualdl
                if (i + 1) % self.args.logging_steps == 0:
                    # log to directory
                    tag_value = {"train/loss": loss.item()}
                    self.visual_writer.update_scalars(tag_value=tag_value, step=self.args.logging_steps)

        time_ed = time.time() - time_st
        print('Train\tLoss: {:.6f}; Time: {:.4f}s'.format(loss.item(), time_ed))

    def eval(self, epoch):
        """eval function

        Args:
            epoch (int): current epoch
        """
        self.model.eval()
        time_st = time.time()

        with tqdm(total=len(self.eval_dataset), disable=True) as pbar:
            names_dataset, outputs_dataset, labels_dataset, attn_dataset = [], [], [], []
            for i, data in enumerate(self.eval_dataloader):
                names = data["names"]
                tokens = data["tokens"]
                input_ids = data["input_ids"]
                labels = data["labels"]

                with paddle.no_grad():
                    output, attn = self.model(tokens, input_ids)

                names_dataset += names
                outputs_dataset.append(output)
                labels_dataset.append(labels)
                attn_dataset.append(attn)
                pbar.update(self.args.batch_size)

            outputs_dataset = paddle.concat(outputs_dataset, axis=0)
            labels_dataset = paddle.concat(labels_dataset, axis=0)
            # save best model
            metrics_dataset = self.compute_metrics(outputs_dataset, labels_dataset)
            if self.args.save_max and self.args.train:
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
        if self.args.train:
            self.visual_writer.update_scalars(tag_value=tag_value, step=1)

        time_ed = time.time() - time_st
        print(log.format(**results), "; Time: {:.4f}s".format(time_ed))

        # args = {}
        # log = 'Test\t'
        # for k, v in metrics_dataset.items():
        #     log += k + ": {" + k + ":.4f}\t"
        #     args[k] = v
        # print(log.format(**args))

        # attn_dataset = paddle.concat(attn_dataset, axis=0)
        # return names_dataset, outputs_dataset, labels_dataset, attn_dataset
