"""
This module builds secondary structure prediction.

Author: wangning(wangning.roci@gmail.com)
Date  : 2023/2/25 15:08
"""

import math
import time
import pickle
import os.path as osp
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import paddle
import paddle.nn as nn
from paddle.nn import functional as F
from paddle.io import Dataset
# from paddle.fluid.reader import DataLoader

from dataset_utils import seq2input_ids
from base_classes import (
    BaseInstance,
    BaseMetrics,
    BaseCollator,
    MlpProjector,
    BaseTrainer
)
import interface


def bpseq2dotbracket(bpseq):
    """convert bpseq to dotbracket

    Args:
        bpseq (str): bpseq file

    Returns:
        str: secondary structure in dotbracket
    """
    dotbracket = []
    for i, x in enumerate(bpseq):
        if x == 0:
            dotbracket.append('.')
        elif x > i:
            dotbracket.append('(')
        else:
            dotbracket.append(')')
    return ''.join(dotbracket)


class BPseqDataset(Dataset):
    """Loads bpseq file to dataset.
    """

    def __init__(self, data_root, bpseq_list):
        """init function

        Args:
            data_root (str): root path to bpseq file
            bpseq_list (str): .lst file to bpseq name
        """
        super().__init__()
        self.data_root = data_root
        with open(bpseq_list) as f:
            self.file_path = f.readlines()
            self.file_path = [x.replace("\n", "") for x in self.file_path]

    def __len__(self):
        """return length of dataset
        """
        return len(self.file_path)

    def __getitem__(self, idx):
        """get item by index

        Args:
            idx (int): item index

        Returns:
            dict: dict contains name, seq and pairs
        """
        file_path = osp.join(self.data_root, self.file_path[idx])
        return self.load_bpseq(file_path)

    def load_bpseq(self, filename):
        """load bpseq file

        Args:
            filename (str): path to bpseq file

        Returns:
            dict: dict contains name, seq and pairs
        """
        with open(filename) as f:
            p = [0]
            s = ['']
            for line in f:
                line = line.rstrip('\n').split()
                idx, c, pair = line
                idx, pair = int(idx), int(pair)
                s.append(c)
                p.append(pair)

        seq = ''.join(s)
        return {"name": filename, "seq": seq, "pairs": np.array(p)}


class SspInstance(BaseInstance):
    """A single instance for sequence classification.
    """

    def __init__(self, name, seq, input_ids, label):
        """init instance

        Args:
            name (str): name of sequence
            seq (str): RNA sequence
            input_ids (int): list of token ids
            label (np.array): contact map of sequence
        """
        super(SspInstance, self).__init__()
        self.name = name
        self.seq = seq
        self.input_ids = input_ids
        self.label = label
        self.length = len(input_ids)


def convert_instance_to_ssp(raw_data, tokenizer):
    """Convert raw data to SspInstance.

    Args:
        raw_data (dict): dict contains name, seq and pairs
        tokenizer (NUCTokenizer): convert sequence to token ids

    Returns:
        SspInstance: instance for secondary structure prediction
    """
    name = raw_data["name"] if "name" in raw_data else None
    pairs = raw_data["pairs"] if "pairs" in raw_data else None
    seq = raw_data["seq"]
    input_ids = seq2input_ids(seq.upper().replace("U", "T"), tokenizer)[:-1]

    return SspInstance(name=name, seq=seq, input_ids=input_ids, label=pairs)


class SspCollator(BaseCollator):
    """Collator for secondary structure prediction.
    """

    def __init__(self):
        """init collator
        """
        super(SspCollator, self).__init__()

    def __call__(self, data):
        """call function of collator

        Args:
            data (SspInstance): secondary structure prediction instance

        Returns:
            dict: dict contains name, seq, input_ids and labels for batch
        """
        instance = data[0]
        name_stack = [getattr(instance, "name")]
        seq_stack = [getattr(instance, "seq")]
        input_ids_stack = getattr(instance, "input_ids")
        labels_stack = getattr(instance, "label")
        return {
            "names": name_stack,
            "seqs": seq_stack,
            "input_ids": self.stack_fn(input_ids_stack),
            "labels": self.stack_fn(labels_stack),
        }


class SparseEmbedding(nn.Layer):
    """Token embedding layer for sparse input.
    """

    def __init__(self, dim):
        """init function

        Args:
            dim (int): embedding dimension
        """
        super(SparseEmbedding, self).__init__()
        self.n_out = dim
        self.embedding = nn.Embedding(6, dim, padding_idx=0)
        self.vocb = defaultdict(
            lambda: 5, {'0': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4})

    def __call__(self, seq):
        """call function
        """
        seq = paddle.to_tensor([[self.vocb[c] for c in s]
                               for s in seq], dtype='int64')
        output = self.embedding(seq)
        output = paddle.transpose(output, perm=[0, 2, 1])
        return output


class CNNLayer(nn.Layer):
    """CNN layer for embedding process.
    """

    def __init__(self,
                 n_in,
                 num_filters=(128, ),
                 filter_size=(7, ),
                 pool_size=(1, ),
                 dilation=1,
                 dropout_rate=0.0,
                 resnet=False):
        """init function

        Args:
            n_in (int): input dimension
            num_filters (tuple, optional): number of filters. Defaults to (128, ).
            filter_size (tuple, optional): kernel size of filters. Defaults to (7, ).
            pool_size (tuple, optional): pooling size. Defaults to (1, ).
            dilation (int, optional): dilation size. Defaults to 1.
            dropout_rate (float, optional): drop out rate. Defaults to 0.0.
            resnet (bool, optional): whether use resnet. Defaults to False.
        """
        super(CNNLayer, self).__init__()
        self.resnet = resnet
        self.net = nn.LayerList()
        for n_out, ksize, p in zip(num_filters, filter_size, pool_size):
            self.net.append(
                nn.Sequential(
                    nn.Conv1D(n_in, n_out, kernel_size=ksize, dilation=2 **
                              dilation, padding=2**dilation * (ksize // 2)),
                    nn.MaxPool1D(p, stride=1, padding=p //
                                 2) if p > 1 else nn.Identity(),
                    nn.GroupNorm(1, n_out),  # same as LayerNorm?
                    nn.CELU(),
                    nn.Dropout(p=dropout_rate)))
            n_in = n_out

    def forward(self, x):  # (B=1, 4, N)
        """forward function

        Args:
            x (paddle.tensor): input tensor

        Returns:
            paddle.tensor: output tensor
        """
        for net in self.net:
            x_a = net(x)
            x = x + x_a if self.resnet and x.shape[1] == x_a.shape[1] else x_a
        return x


class CNNLSTMEncoder(nn.Layer):
    """CNN LSTM encoder for embedding process.
    """

    def __init__(self,
                 n_in,
                 num_filters=(256, ),
                 filter_size=(7, ),
                 pool_size=(1, ),
                 dilation=0,
                 num_lstm_layers=0,
                 num_lstm_units=0,
                 num_att=0,
                 dropout_rate=0.0,
                 resnet=True):
        """init function

        Args:
            n_in (int): input dimension
            num_filters (tuple, optional): number of filters. Defaults to (256, ).
            filter_size (tuple, optional): kernel size of filters. Defaults to (7, ).
            pool_size (tuple, optional): pool size. Defaults to (1, ).
            dilation (int, optional): dilation number. Defaults to 0.
            num_lstm_layers (int, optional): number of lstm layers. Defaults to 0.
            num_lstm_units (int, optional): mumber of lstm units. Defaults to 0.
            num_att (int, optional): number of attention head. Defaults to 0.
            dropout_rate (float, optional): drop out rate. Defaults to 0.0.
            resnet (bool, optional): whether use resnet. Defaults to True.
        """
        super(CNNLSTMEncoder, self).__init__()
        self.resnet = resnet
        self.n_in = self.n_out = n_in
        while len(num_filters) > len(filter_size):
            filter_size = tuple(filter_size) + (filter_size[-1], )
        while len(num_filters) > len(pool_size):
            pool_size = tuple(pool_size) + (pool_size[-1], )
        if num_lstm_layers == 0 and num_lstm_units > 0:
            num_lstm_layers = 1

        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv = self.lstm = self.att = None

        if len(num_filters) > 0 and num_filters[0] > 0:
            self.conv = CNNLayer(n_in,
                                 num_filters,
                                 filter_size,
                                 pool_size,
                                 dilation,
                                 dropout_rate=dropout_rate,
                                 resnet=self.resnet)
            self.n_out = n_in = num_filters[-1]

        if num_lstm_layers > 0:
            self.lstm = nn.LSTM(
                n_in,
                num_lstm_units,
                num_layers=num_lstm_layers,
                # batch_first=True,
                direction='bidirectional',
                dropout=dropout_rate if num_lstm_layers > 1 else 0)
            self.n_out = num_lstm_units * 2
            self.lstm_ln = nn.LayerNorm(self.n_out)

        if num_att > 0:
            self.att = paddle.nn.MultiHeadAttention(
                self.n_out, num_att, dropout=dropout_rate)

    def forward(self, x):  # (B, n_in, N)
        """forward function

        Args:
            x (paddle.tensor): input tensor

        Returns:
            paddle.tensor: output tensor
        """
        if self.conv is not None:
            x = self.conv(x)  # (B, C, N)
        x = paddle.transpose(x, perm=[0, 2, 1])  # (B, N, C)

        if self.lstm is not None:
            x_a, _ = self.lstm(x)
            x_a = self.lstm_ln(x_a)
            x_a = self.dropout(F.celu(x_a))  # (B, N, H*2)
            x = x + x_a if self.resnet and x.shape[2] == x_a.shape[2] else x_a

        if self.att is not None:
            x = paddle.transpose(x, perm=[1, 0, 2])
            x_a = self.att(x, x, x)
            x = x + x_a
            x = paddle.transpose(x, perm=[1, 0, 2])

        return x


class Transform2D(nn.Layer):
    """2D transform layer for 2D attention.
    """

    def __init__(self, join='cat', context_length=0):
        """init function

        Args:
            join (str, optional): how to concat embeddings. Defaults to 'cat'.
            context_length (int, optional): context length. Defaults to 0.
        """
        super(Transform2D, self).__init__()
        self.join = join

    def forward(self, x_l, x_r):
        """forward function

        Args:
            x_l (paddle.tensor): left side tensor
            x_r (paddle.tensor): right side tensor

        Raises:
            NotImplementedError: not implemented error

        Returns:
            paddle.tensor: output tensor
        """
        assert (x_l.shape == x_r.shape)
        B, N, C = x_l.shape
        x_l = x_l.unsqueeze(axis=2).expand(shape=[B, N, N, C])
        x_r = x_r.unsqueeze(axis=1).expand(shape=[B, N, N, C])
        if self.join == 'cat':
            x = paddle.concat((x_l, x_r), axis=3)  # (B, N, N, C*2)
        elif self.join == 'add':
            x = x_l + x_r  # (B, N, N, C)
        elif self.join == 'mul':
            x = x_l * x_r  # (B, N, N, C)
        else:
            raise NotImplementedError
        return x


class PairedLayer(nn.Layer):
    """paired layer for 2D attention.
    """

    def __init__(self, n_in, n_out=1, filters=(), ksize=(), fc_layers=(), dropout_rate=0.0, exclude_diag=True, resnet=True):
        """init function

        Args:
            n_in (int): input dimension
            n_out (int, optional): output dimension. Defaults to 1.
            filters (tuple, optional): number of filters. Defaults to ().
            ksize (tuple, optional): kernel size of filters. Defaults to ().
            fc_layers (tuple, optional): fully connected layers. Defaults to ().
            dropout_rate (float, optional): drop out rate. Defaults to 0.0.
            exclude_diag (bool, optional): whether exclude diagonal elements. Defaults to True.
            resnet (bool, optional): whether use resnet. Defaults to True.
        """
        super(PairedLayer, self).__init__()

        self.resnet = resnet
        self.exclude_diag = exclude_diag
        while len(filters) > len(ksize):
            ksize = tuple(ksize) + (ksize[-1], )

        self.conv = nn.LayerList()
        for m, k in zip(filters, ksize):
            self.conv.append(
                nn.Sequential(nn.Conv2D(n_in, m, k, padding=k // 2), nn.GroupNorm(1, m), nn.CELU(), nn.Dropout(p=dropout_rate)))
            n_in = m

        fc = []
        for m in fc_layers:
            fc += [nn.Linear(n_in, m), nn.LayerNorm(m),
                   nn.CELU(), nn.Dropout(p=dropout_rate)]
            n_in = m
        fc += [nn.Linear(n_in, n_out)]
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        """forward function

        Args:
            x (paddle.tensor): input tensor

        Returns:
            paddle.tensor: output tensor
        """
        diag = 1 if self.exclude_diag else 0
        B, N, _, C = x.shape
        x = paddle.transpose(x, perm=[0, 3, 1, 2])
        x_u = paddle.reshape(paddle.triu(paddle.reshape(
            x, shape=[B * C, N, N]), diagonal=diag), shape=[B, C, N, N])
        x_l = paddle.reshape(paddle.tril(paddle.reshape(
            x, shape=[B * C, N, N]), diagonal=-1), shape=[B, C, N, N])
        x = paddle.reshape(paddle.concat(
            [x_u, x_l], axis=0), shape=[B * 2, C, N, N])
        for conv in self.conv:
            x_a = conv(x)
            # (B*2, n_out, N, N)
            x = x + x_a if self.resnet and x.shape[1] == x_a.shape[1] else x_a
        x_u, x_l = paddle.split(x, num_or_sections=2,
                                axis=0)  # (B, n_out, N, N) * 2
        x_u = paddle.triu(paddle.reshape(
            x_u, shape=[B, -1, N, N]), diagonal=diag)
        x_l = paddle.tril(paddle.reshape(
            x_u, shape=[B, -1, N, N]), diagonal=-1)
        x = x_u + x_l  # (B, n_out, N, N)
        x = paddle.reshape(paddle.transpose(
            x, perm=[0, 2, 3, 1]), shape=[B * N * N, -1])
        x = self.fc(x)
        x = paddle.reshape(x, shape=[B, N, N, -1])
        return x


class NeuralNet(nn.Layer):
    """Neural network for secondary structure prediction.
    """

    def __init__(self,
                 embed_size=0,
                 num_filters=(96, ),
                 filter_size=(5, ),
                 dilation=0,
                 pool_size=(1, ),
                 num_lstm_layers=0,
                 num_lstm_units=0,
                 num_att=0,
                 no_split_lr=False,
                 pair_join='cat',
                 num_paired_filters=(),
                 paired_filter_size=(),
                 num_hidden_units=(32, ),
                 dropout_rate=0.0,
                 fc_dropout_rate=0.0,
                 exclude_diag=True,
                 n_out_paired_layers=0,
                 n_out_unpaired_layers=0,
                 num_transformer_layers=0):
        """init function

        Args:
            embed_size (int, optional): embedding size. Defaults to 0.
            number_filters (tuple, optional): number of filters. Defaults to (96, ).
            filer_size (tuple, optional): filter size. Defaults to (5, ).
            dilation (int, optional): dilation. Defaults to 0.
            pool_size (tuple, optional): pool size. Defaults to (1, ).
            num_lstm_layers (int, optional): number of lstm layers. Defaults to 0.
            num_lstm_units (int, optional): number of lstm units. Defaults to 0.
            num_att (int, optional): number of attention layers. Defaults to 0.
            no_split_lr (bool, optional): whether split tensor. Defaults to False.
            pair_join (str, optional): how to join paired tensors. Defaults to 'cat'.
            num_paired_filters (tuple, optional): number of paired filters. Defaults to ().
            paried_filter_size (tuple, optional): paired filter size. Defaults to ().
            num_hidden_units (tuple, optional): number of hidden units. Defaults to (32, ).
            dropout_rate (float, optional): drop out rate. Defaults to 0.0.
            fc_dropout_rate (float, optional): fully connected layer drop out rate. Defaults to 0.0.
            exclude_diag (bool, optional): whether exclude diagonal. Defaults to True.
            n_out_paired_layers (int, optional): number of output paired layers. Defaults to 0.
            n_out_unpaired_layers (int, optional): number of output unpaired layers. Defaults to 0.
            num_transformer_layers (int, optional): number of transformer layers. Defaults to 0.
        """

        super(NeuralNet, self).__init__()

        self.no_split_lr = no_split_lr
        self.pair_join = pair_join
        self.embedding = SparseEmbedding(embed_size)
        n_in = self.embedding.n_out

        self.proj_head = MlpProjector(768, 128)
        self.encoder = CNNLSTMEncoder(n_in + 128,
                                      num_filters=num_filters,
                                      filter_size=filter_size,
                                      pool_size=pool_size,
                                      dilation=dilation,
                                      num_att=num_att,
                                      num_lstm_layers=num_lstm_layers,
                                      num_lstm_units=num_lstm_units,
                                      dropout_rate=dropout_rate)
        n_in = self.encoder.n_out

        self.transform2d = Transform2D(join=pair_join)

        n_in_paired = n_in // 2 if pair_join != 'cat' else n_in

        self.fc_paired = PairedLayer(n_in_paired,
                                     n_out_paired_layers,
                                     filters=num_paired_filters,
                                     ksize=paired_filter_size,
                                     exclude_diag=exclude_diag,
                                     fc_layers=num_hidden_units,
                                     dropout_rate=fc_dropout_rate)
        self.fc_unpaired = None

    def forward(self, seq, embeddings):
        """forward function

        Args:
            seq (str): raw RNA sequence
            embeddings (paddle.tensor): pretrained RNAErine embeddings

        Returns:
            (paddle.tensor, paddle.tensor): paired and unpaired scores
        """
        x = self.embedding(['0' + s for s in seq])  # (B, 4, N)
        embeddings = self.proj_head(embeddings)

        x = paddle.concat([x, paddle.transpose(
            embeddings, perm=[0, 2, 1])], axis=1)
        x = self.encoder(x)

        # if self.no_split_lr:
        #     x_l, x_r = x, x
        # else:
        x_l = x[:, :, 0::2]
        x_r = x[:, :, 1::2]
        x_r = paddle.fluid.layers.reverse(x_r, axis=-1)
        x_lr = self.transform2d(x_l, x_r)

        score_paired = self.fc_paired(x_lr)
        if self.fc_unpaired is not None:
            score_unpaired = self.fc_unpaired(x)
        else:
            score_unpaired = None

        return score_paired, score_unpaired


class LengthLayer(nn.Layer):
    """Length prediction layer.
    """

    def __init__(self, n_in, layers=(), dropout_rate=0.5):
        """init function

        Args:
            n_in (int): input dimension.
            layers (tuple, optional): layers. Defaults to ().
            dropout_rate (float, optional): dropout rate. Defaults to 0.5.
        """
        super(LengthLayer, self).__init__()
        self.n_in = n_in
        n = n_in if isinstance(n_in, int) else np.prod(n_in)

        layer = []
        for m in layers:
            layer += [nn.Linear(n, m), nn.CELU(), nn.Dropout(p=dropout_rate)]
            n = m
        layer += [nn.Linear(n, 1)]
        self.net = nn.Sequential(*layer)

        if isinstance(self.n_in, int):
            self.x = paddle.tril(paddle.ones((self.n_in, self.n_in)))
        else:
            n = np.prod(self.n_in)
            x = np.fromfunction(lambda i, j, k, ll: np.logical_and(
                k <= i, ll <= j), (*self.n_in, *self.n_in))
            self.x = paddle.reshape(paddle.to_tensor(
                x.astype(np.int64), dtype="float32"), shape=[n, n])

    def forward(self, x):
        """forward function

        Args:
            x (paddle.tensor): input tensor

        Returns:
            paddle.tensor: output tensor
        """
        return self.net(x)

    def make_param(self):
        """make parameter
        """
        x = self.forward(self.x)
        return x.reshape((self.n_in, ) if isinstance(self.n_in, int) else self.n_in)


class AbstractFold(nn.Layer):
    """Abstract fold model.
    """

    def __init__(self, predict):
        """init function

        Args:
            predict (interface): theromodynamic prediction interface
        """
        super(AbstractFold, self).__init__()
        self.predict = predict

    def clear_count(self, param):
        """clear count

        Args:
            param (paddle.tensor): model parameters

        Returns:
            paddle.tensor: parameters with count
        """
        param_count = {}
        for n, p in param.items():
            if n.startswith("score_"):
                param_count["count_" + n[6:]] = paddle.zeros_like(p).detach()
        param.update(param_count)
        return param

    def clear_count_cpp(self, param):
        """clear cpp count

        Args:
            param (np.array): model parameters

        Returns:
            np.array: parameters with count
        """
        param_count = {}
        for n, p in param.items():
            if n.startswith("score_"):
                param_count["count_" + n[6:]] = np.zeros_like(p)
        param.update(param_count)
        return param

    def calculate_differentiable_score(self, v, param, count):
        """calculate differentiable score

        Args:
            v (paddle.tensor): value of thermodynamic score
            param (paddle.tensor): model parameters
            count (int): parameter count

        Returns:
            paddle.tensor: score with gradient
        """
        s = 0
        for n, p in param.items():
            if n.startswith("score_"):
                s += paddle.sum(p * count["count_" + n[6:]])
        s += v - s.detach()
        return s

    def forward(self,
                seq,
                return_param=False,
                param=None,
                max_internal_length=30,
                max_helix_length=30,
                constraint=None,
                reference=None,
                loss_pos_paired=0.0,
                loss_neg_paired=0.0,
                loss_pos_unpaired=0.0,
                loss_neg_unpaired=0.0):
        """forward function

        Args:
            seq (str): RNA sequence.
            return_param (bool, optional): whether return parameters. Defaults to False.
            param (_type_, optional): _description_. Defaults to None.
            max_internal_length (int, optional): _description_. Defaults to 30.
            max_helix_length (int, optional): _description_. Defaults to 30.
            constraint (_type_, optional): _description_. Defaults to None.
            reference (_type_, optional): _description_. Defaults to None.
            loss_pos_paired (float, optional): _description_. Defaults to 0.0.
            loss_neg_paired (float, optional): _description_. Defaults to 0.0.
            loss_pos_unpaired (float, optional): _description_. Defaults to 0.0.
            loss_neg_unpaired (float, optional): _description_. Defaults to 0.0.

        Returns:
            tuple: secondary structure, prediction, pairs, parameters
        """
        param = self.make_param(
            seq) if param is None else param  # reuse param or not
        ss = []
        preds = []
        pairs = []
        for i in range(len(seq)):
            param_on_cpu_cpp = {k: v.numpy() for k, v in param[i].items()}

            with paddle.no_grad():
                v, pred, pair = self.predict(
                    seq[i],
                    self.clear_count_cpp(param_on_cpu_cpp),
                    max_internal_length=max_internal_length if max_internal_length is not None else len(
                        seq[i]),
                    max_helix_length=max_helix_length,
                    constraint=constraint[i].tolist(
                    ) if constraint is not None else None,
                    reference=reference[i].tolist(
                    ) if reference is not None else None,
                    loss_pos_paired=loss_pos_paired,
                    loss_neg_paired=loss_neg_paired,
                    loss_pos_unpaired=loss_pos_unpaired,
                    loss_neg_unpaired=loss_neg_unpaired)
            ss.append(v)
            preds.append(pred)
            pairs.append(pair)

        ss = paddle.stack(ss) if paddle.is_grad_enabled(
        ) else paddle.to_tensor(ss)
        if return_param:
            return ss, preds, pairs, param
        else:
            return ss, preds, pairs


class ZukerFold(AbstractFold):
    """Zuker fold model.
    """

    def __init__(self, max_helix_length=30, **kwargs):
        """init function

        Args:
            max_helix_length (int, optional): max helix length. Defaults to 30.
        """
        super(ZukerFold, self).__init__(predict=interface.predict_zuker)

        self.max_helix_length = max_helix_length
        self.net = NeuralNet(**kwargs)

        self.fc_length = nn.LayerDict({
            'score_hairpin_length': LengthLayer(31),
            'score_bulge_length': LengthLayer(31),
            'score_internal_length': LengthLayer(31),
            'score_internal_explicit': LengthLayer((5, 5)),
            'score_internal_symmetry': LengthLayer(16),
            'score_internal_asymmetry': LengthLayer(29),
            'score_helix_length': LengthLayer(31)
        })

    def make_param(self, seq, embeddings):
        """make parameters

        Args:
            seq (str): RNA sequence.
            embeddings (paddle.tensor): pretrained embeddings of RNA sequence.
        """
        score_paired, score_unpaired = self.net(seq, embeddings)
        B, N, _, _ = score_paired.shape

        score_basepair = paddle.zeros((B, N, N))
        score_helix_stacking = score_paired[:, :, :, 0]  # (B, N, N)
        score_mismatch_external = score_paired[:, :, :, 1]  # (B, N, N)
        score_mismatch_internal = score_paired[:, :, :, 1]  # (B, N, N)
        score_mismatch_multi = score_paired[:, :, :, 1]  # (B, N, N)
        score_mismatch_hairpin = score_paired[:, :, :, 1]  # (B, N, N)
        score_unpaired = score_paired[:, :, :, 2]  # (B, N, N)
        score_base_hairpin = score_unpaired
        score_base_internal = score_unpaired
        score_base_multi = score_unpaired
        score_base_external = score_unpaired

        param = [{
            'score_basepair': score_basepair[i],
            'score_helix_stacking': score_helix_stacking[i],
            'score_mismatch_external': score_mismatch_external[i],
            'score_mismatch_hairpin': score_mismatch_hairpin[i],
            'score_mismatch_internal': score_mismatch_internal[i],
            'score_mismatch_multi': score_mismatch_multi[i],
            'score_base_hairpin': score_base_hairpin[i],
            'score_base_internal': score_base_internal[i],
            'score_base_multi': score_base_multi[i],
            'score_base_external': score_base_external[i],
            'score_hairpin_length': self.fc_length['score_hairpin_length'].make_param(),
            'score_bulge_length': self.fc_length['score_bulge_length'].make_param(),
            'score_internal_length': self.fc_length['score_internal_length'].make_param(),
            'score_internal_explicit': self.fc_length['score_internal_explicit'].make_param(),
            'score_internal_symmetry': self.fc_length['score_internal_symmetry'].make_param(),
            'score_internal_asymmetry': self.fc_length['score_internal_asymmetry'].make_param(),
            'score_helix_length': self.fc_length['score_helix_length'].make_param()
        } for i in range(B)]

        return param


class RNAFold(AbstractFold):
    """RNA Fold model.
    """

    def __init__(self, init_param=None):
        """init function

        Args:
            init_param (np.array, optional): init parameters. Defaults to None.
        """
        super(RNAFold, self).__init__(interface.predict_turner)

        for n in dir(init_param):
            if n.startswith("score_"):
                value = getattr(init_param, n)

                setattr(self, n, paddle.create_parameter(
                    shape=value.shape, dtype='float32'))
                getattr(self, n).set_value(
                    paddle.to_tensor(value, stop_gradient=False))

    def make_param(self, seq):
        """make parameters

        Args:
            seq (str): RNA sequence.
        """
        param = {n: getattr(self, n)
                 for n in dir(self) if n.startswith("score_")}
        # print(param)
        return [param for s in seq]


class MixedFold(AbstractFold):
    """Mixed fold model.
    """

    def __init__(self, init_param=None, max_helix_length=30, **kwargs):
        """init function
        """
        super(MixedFold, self).__init__(interface.predict_mxfold)
        self.turner = RNAFold(init_param=init_param)
        self.zuker = ZukerFold(max_helix_length=max_helix_length, **kwargs)
        self.max_helix_length = max_helix_length

    def forward(self,
                seq,
                embeddings,
                return_param=False,
                param=None,
                return_partfunc=False,
                max_internal_length=30,
                constraint=None,
                reference=None,
                loss_pos_paired=0.0,
                loss_neg_paired=0.0,
                loss_pos_unpaired=0.0,
                loss_neg_unpaired=0.0):
        """Forward function.

        Args:
            seq: RNA sequence.
            embeddings: Embeddings of pretrained RNA sequence.
            return_param: Return param or not.
            param: Model parameters.
            return_partfunc: Return partition function or not.
            max_internal_length: Max internal length.
            constraint: Constraint of base pair.
            reference: Reference structure.
            loss_pos_paired: Loss of positive paired.
            loss_neg_paired: Loss of negative paired.
            loss_pos_unpaired: Loss of positive unpaired.
            loss_neg_unpaired: Loss of negative unpaired.

        Returns:
            tuple: secondary structure, prediction, pairs, parameters
        """
        param = self.make_param(
            seq, embeddings) if param is None else param  # reuse param or not
        ss = []
        preds = []
        pairs = []
        for i in range(len(seq)):
            param_on_cpu = {
                'turner': {k: v
                           for k, v in param[i]['turner'].items()},
                'positional': {k: v
                               for k, v in param[i]['positional'].items()}
            }
            param_on_cpu_cpp = {
                'turner': {k: v.numpy()
                           for k, v in param[i]['turner'].items()},
                'positional': {k: v.numpy()
                               for k, v in param[i]['positional'].items()}
            }
            param_on_cpu = {k: self.clear_count(
                v) for k, v in param_on_cpu.items()}
            param_on_cpu_cpp = {k: self.clear_count_cpp(
                v) for k, v in param_on_cpu_cpp.items()}

            with paddle.no_grad():
                v1, pred, pair = interface.predict_mxfold(
                    seq[i],
                    param_on_cpu_cpp,
                    max_internal_length=max_internal_length if max_internal_length is not None else len(
                        seq[i]),
                    max_helix_length=self.max_helix_length,
                    constraint=constraint[i].tolist(
                    ) if constraint is not None else None,
                    reference=reference[i].tolist(
                    ) if reference is not None else None,
                    loss_pos_paired=loss_pos_paired,
                    loss_neg_paired=loss_neg_paired,
                    loss_pos_unpaired=loss_pos_unpaired,
                    loss_neg_unpaired=loss_neg_unpaired)
            for k, v in param_on_cpu.items():
                for kk, vv in v.items():
                    param_on_cpu[k][kk] = paddle.to_tensor(
                        param_on_cpu_cpp[k][kk], stop_gradient=False)

            if paddle.is_grad_enabled():
                v1 = self.calculate_differentiable_score(
                    v1, param[i]['positional'], param_on_cpu['positional'])
            ss.append(v1)
            preds.append(pred)
            pairs.append(pair)

        ss = paddle.stack(ss) if paddle.is_grad_enabled(
        ) else paddle.to_tensor(ss)
        if return_param:
            return ss, preds, pairs, param
        else:
            return ss, preds, pairs

    def make_param(self, seq, embeddings):
        """make parameters

        Args:
            seq (str): RNA sequence.
            embeddings (paddle.tensor): Embeddings of pretrained RNA sequence.
        """
        ts = self.turner.make_param(seq)
        ps = self.zuker.make_param(seq, embeddings)
        return [{'turner': t, 'positional': p} for t, p in zip(ts, ps)]


class StructuredLoss(nn.Layer):
    """Structured loss
    """

    def __init__(self,
                 loss_pos_paired=0,
                 loss_neg_paired=0,
                 loss_pos_unpaired=0,
                 loss_neg_unpaired=0,
                 l1_weight=0.,
                 l2_weight=0.):
        """init function

        Args:
            loss_pos_paired (int, optional): positive paired loss. Defaults to 0.
            loss_neg_paired (int, optional): negative paired loss. Defaults to 0.
            loss_pos_unpaired (int, optional): positive unpaired loss. Defaults to 0.
            loss_neg_unpaired (int, optional): negative unpaired loss. Defaults to 0.
            l1_weight (_type_, optional): l1 loss weight. Defaults to 0..
            l2_weight (_type_, optional): l2 loss weight. Defaults to 0..
        """
        super(StructuredLoss, self).__init__()
        # self.model = model
        self.loss_pos_paired = loss_pos_paired
        self.loss_neg_paired = loss_neg_paired
        self.loss_pos_unpaired = loss_pos_unpaired
        self.loss_neg_unpaired = loss_neg_unpaired
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def forward(self, model, seq, pairs, embeddings, fname=None):
        """forward function

        Args:
            model (paddle.nn.Layer): paddle model
            seq (str): raw RNA sequence
            pairs (np.array): base pair constraint
            embeddings (paddle.tensor): pretrained RNA sequence embeddings
            fname (str, optional): RNA sequence name. Defaults to None.

        Returns:
            paddle.tensor: sum of loss
        """
        pred, pred_s, _, param = model(seq,
                                       embeddings,
                                       return_param=True,
                                       reference=pairs,
                                       loss_pos_paired=self.loss_pos_paired,
                                       loss_neg_paired=self.loss_neg_paired,
                                       loss_pos_unpaired=self.loss_pos_unpaired,
                                       loss_neg_unpaired=self.loss_neg_unpaired)
        ref, ref_s, _ = model(seq, embeddings, param=param, constraint=pairs, max_internal_length=None)
        length = paddle.to_tensor([len(s) for s in seq])
        loss = (pred - ref) / length

        if loss.item() > 1e10 or paddle.isnan(loss):
            print()
            print(fname)
            print(loss.item(), pred.item(), ref.item())
            print(seq)

        if self.l1_weight > 0.0:
            for p in model.parameters():
                loss += self.l1_weight * paddle.sum(paddle.abs(p))
        return paddle.sum(loss)


def compare_bpseq(ref, pred):
    """compare bpseq reference and prediction

    Args:
        ref (np.array): bpseq reference
        pred (np.array): predicted

    Returns:
        tuple: true positive, true negative, false positive, false negative
    """
    L = len(ref) - 1
    tp = fp = fn = 0
    if (len(ref) > 0 and isinstance(ref[0], list)) or (isinstance(ref, paddle.Tensor) and ref.ndim == 2):
        if isinstance(ref, paddle.Tensor):
            ref = ref.tolist()
        ref = {(min(i, j), max(i, j)) for i, j in ref}
        pred = {(i, j) for i, j in enumerate(pred) if i < j}
        tp = len(ref & pred)
        fp = len(pred - ref)
        fn = len(ref - pred)
    else:
        assert (len(ref) == len(pred))
        for i, (j1, j2) in enumerate(zip(ref, pred)):
            if j1 > 0 and i < j1:  # pos
                if j1 == j2:
                    tp += 1
                elif j2 > 0 and i < j2:
                    fp += 1
                    fn += 1
                else:
                    fn += 1
            elif j2 > 0 and i < j2:
                fp += 1
    tn = L * (L - 1) // 2 - tp - fp - fn
    return tp, tn, fp, fn


class SspMetric(BaseMetrics):
    """Secondary structure prediction metric.
    """

    def __init__(self, metrics):
        """init function
        """
        super(SspMetric, self).__init__(metrics=metrics)

    def __call__(self, tp, tn, fp, fn):
        """calculate metric

        Args:
            tp (int): true positive.
            tn (int): true negative.
            fp (int): false positive.
            fn (int): false negative.

        Returns:
            dict: dict of metric.
        """
        ret = {}
        acc = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0. else 0.
        sen = tp / (tp + fn) if tp + fn > 0. else 0.
        ppv = tp / (tp + fp) if tp + fp > 0. else 0.
        fval = 2 * sen * ppv / (sen + ppv) if sen + ppv > 0. else 0.
        mcc = ((tp * tn) - (fp * fn)) / math.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0. else 0.
        for m in self.metrics:
            if m == "accuracy":
                ret[m] = acc
            elif m == "recall":
                ret[m] = sen
            elif m == "precision":
                ret[m] = ppv
            elif m == "f1s":
                ret[m] = fval
            elif m == "mcc":
                ret[m] = mcc
        return ret


class SspTrainer(BaseTrainer):
    """Secondary structure prediction trainer.
    """

    def __init__(self,
                 args,
                 tokenizer,
                 model,
                 pretrained_model=None,
                 indicator=None,
                 train_dataset=None,
                 eval_dataset=None,
                 data_collator=None,
                 loss_fn=None,
                 optimizer=None,
                 compute_metrics=None,
                 visual_writer=None):
        """init function

        Args:
            args (argparse): args from argparse
            tokenizer (NUCTokenizer): tokenizer to encode the input sequence
            model (MixedFold): RNA secondary structure prediction model
            pretrained_model (str, optional): pretrained model. Defaults to None.
            indicator (BaseIndicator, optional): indicator to show training process. Defaults to None.
            train_dataset (BPseqDataset): training dataset
            eval_dataset (BPseqDataset, optional): eval dataset. Defaults to None.
            data_collator (SspCollator, optional): data collator. Defaults to None.
            loss_fn (StructuredLoss, optional): loss function. Defaults to None.
            optimizer (paddle.optimizer, optional): parameters optimizer. Defaults to None.
            compute_metrics (SspMetric, optional): how to compute metrics. Defaults to None.
            visual_writer (visualizer.Visualizer, optional): visualdl writer. Defaults to None.
        """
        super(SspTrainer, self).__init__(
            args=args,
            tokenizer=tokenizer,
            model=model,
            pretrained_model=pretrained_model,
            indicator=indicator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            loss_fn=loss_fn,
            optimizer=optimizer,
            compute_metrics=compute_metrics,
            visual_writer=visual_writer
        )
        self.is_break = False

    def get_status(self):
        """get training status

        Returns:
            boolean: whether to break training
        """
        return self.is_break

    def train(self, epoch):
        """train model

        Args:
            epoch: training epoch

        Returns:
            None
        """

        self.model.train()
        time_st = time.time()
        loss_total, num_total = 0, 0

        with tqdm(total=len(self.train_dataset), disable=self.args.disable_tqdm) as pbar:
            for i, instance in enumerate(self.train_dataloader):
                headers = instance["names"]
                refs = (instance["labels"], )
                seqs = (instance["seqs"][0], )
                input_ids = (instance["input_ids"], )

                if instance["input_ids"].shape[0] < 600:
                    with paddle.no_grad():
                        input_tensor = paddle.to_tensor(input_ids[0]).unsqueeze(0)
                        embeddings = self.pretrained_model(input_tensor)[0].detach()
                    self.optimizer.clear_grad()
                    loss = self.loss_fn(self.model, seqs, refs, embeddings, fname=headers)
                    if loss.item() > 0.:
                        loss.backward()
                        self.optimizer.step()
                    if paddle.isnan(loss):
                        self.is_break = True
                        break
                else:
                    print(headers)
                break
                # log to pbar
                num_total += self.args.batch_size
                loss_total += loss.item()
                pbar.set_postfix(train_loss='{:.3e}'.format(loss_total / num_total))
                pbar.update(self.args.batch_size)
                # reset loss if too many steps
                if num_total >= self.args.logging_steps:
                    loss_total, num_total = 0, 0
                # log to visualdl
                if (i + 1) % self.args.logging_steps:
                    # log to directory
                    tag_value = {"train/loss": loss.item()}
                    self.visual_writer.update_scalars(tag_value=tag_value, step=1)

        elapsed_time = time.time() - time_st
        print('Train Epoch: {}\tTime: {:.3f}s'.format(epoch, elapsed_time))

    def eval(self, epoch, save_path=None):
        """
        Args:
            epoch: eval epoch

        Returns:
            None
        """
        self.model.eval()
        n_dataset = len(self.eval_dataloader.dataset)
        with tqdm(total=n_dataset) as pbar:
            # init with default values
            res = defaultdict(list)
            ret_dataset, names_dataset, lens_dataet, scs_dataset, preds_dataset = [], [], [], [], []
            for instance in self.eval_dataloader:
                headers = instance["names"]
                refs = (instance["labels"], )
                seqs = (instance["seqs"][0], )
                input_ids = (instance["input_ids"], )
                seqs = (seqs[0], )
                refs = (refs[0], )
                if instance["input_ids"].shape[0] < 600:
                    if self.args.two_stage:
                        input_ids_ind = paddle.concat([input_ids[0], paddle.to_tensor([0, 0, 0])], axis=0)
                        input_ids_inds, topk_probs, _ = self.get_input_ids_ind_topk(
                            input_ids=input_ids_ind,
                            seq_lens=paddle.to_tensor(len(input_ids_ind)),
                            indicator=self.indicator
                        )
                        input_ids_inds = input_ids_inds.squeeze(0)
                        embeddings = self.pretrained_model(input_ids_inds)[0].detach()
                        topk_probs = paddle.squeeze(topk_probs, axis=0)
                        embeddings = paddle.einsum('ijk,i->jk', embeddings, topk_probs)
                        embeddings = embeddings[:-3, :]
                        scs, preds, bps = self.model(seqs, embeddings.unsqueeze(0))
                    else:
                        with paddle.no_grad():
                            input_tensor = paddle.to_tensor(input_ids[0]).unsqueeze(0)
                            embeddings = self.pretrained_model(input_tensor)[0].detach()
                        scs, preds, bps = self.model(seqs, embeddings)
                    for header, seq, ref, sc, pred, bp in zip(headers, seqs, refs, scs, preds, bps):
                        x = compare_bpseq(ref, bp)
                        ret = self.compute_metrics(*x)
                        for k, v in ret.items():
                            res[k].append(v)

                if save_path:
                    ret_dataset.append(ret)
                    names_dataset.append(headers[0])
                    lens_dataet.append(len(seqs[0]))
                    scs_dataset.append(scs[0].item())
                    preds_dataset.append(preds[0])

                pbar.set_postfix(eval_fls='{:.3e}'.format(np.mean(res[self.name_pbar])))
                pbar.update(self.args.batch_size)

            # save best model
            metrics_dataset = {}
            for k, v in res.items():
                metrics_dataset[k] = np.mean(v)
            if self.args.save_max and self.args.train:
                self.save_model(metrics_dataset, epoch)
            # log results to screen/bash
            results = {}
            log = 'Test\t' + self.args.task_name + "\t"
            # log results to visualdl
            tag_value = defaultdict(float)
            # extract results
            for k, v in metrics_dataset.items():
                log += k + ": {" + k + ":.4f}\t"
                results[k] = v
                tag = "eval/" + k
                tag_value[tag] = v
            print(log.format(**results))
            if self.visual_writer:
                self.visual_writer.update_scalars(tag_value=tag_value, step=1)
            if save_path:
                # save results to pickle file
                results = {
                    "ret_dataset": ret_dataset,
                    "names_dataset": names_dataset,
                    "lens_dataet": lens_dataet,
                    "scs_dataset": scs_dataset,
                    "preds_dataset": preds_dataset
                }
                with open(save_path, 'wb') as f:
                    pickle.dump(results, f)

    def predict(self, batch_converter, data):
        self.model.eval()
        with paddle.no_grad():
            # show progress bar
            with tqdm(total=len(data)) as pbar:
                for names, seqs, inputs_ids in batch_converter(data):
                    print(inputs_ids)
                    seqs = [x.replace("T", "U") for x in seqs]
                    inputs_ids = inputs_ids[:, :-1]
                    with paddle.no_grad():
                        embeddings = self.pretrained_model(inputs_ids)[0].detach()
                    scs, preds, bps = self.model(seqs, embeddings)
                    print(names)
                    print(seqs)
                    print(scs)
                    print(preds)

                    pbar.update(1)
