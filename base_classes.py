"""
This module builds base trainer for all pre & downstream tasks.

Author: wangning(wangning.roci@gmail.com)
Date  : 2022/12/8 2:43 PM
"""
import copy
import os
import shutil
import numpy as np
import abc
import os.path as osp

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.fluid.reader import DataLoader, BatchSampler
from paddlenlp.data import Stack
from paddlenlp.transformers import ErnieForMaskedLM


# ===================== Common Modules =====================
class MlpProjector(nn.Layer):
    """MLP projection head.
    """

    def __init__(self, n_in, output_size=128):
        """
        Args:
            n_in:
            output_size:
        """
        super(MlpProjector, self).__init__()
        self.dense = nn.Linear(n_in, output_size)
        self.activation = nn.ReLU()
        self.projection = nn.Linear(output_size, output_size)
        self.n_out = output_size

    def forward(self, embeddings):
        """
        Args:
            embeddings:

        Returns:

        """
        x = self.dense(embeddings)
        x = self.activation(x)
        x = self.projection(x)

        return x


class IndicatorClassifier(nn.Layer):
    """This class indicate coarse class after token [IND].
    """

    def __init__(self, model_name_or_path):
        """
        Args:
            model_name_or_path:
        """
        super(IndicatorClassifier, self).__init__()
        self.indicator = ErnieForMaskedLM.from_pretrained(model_name_or_path)

    def forward(self,
                input_ids,
                masked_positions):
        """
        Args:
            input_ids:
            masked_positions:
        Returns:

        """
        with paddle.no_grad():
            outputs = self.indicator(
                input_ids, masked_positions=masked_positions)
            return outputs


class MajorityVoter(nn.Layer):
    """Stack top k logits.
    """

    def __init__(self):
        """
        """
        super(MajorityVoter, self).__init__()

    def forward(self, k_logits, topk_probs):
        """
        Forward function.

        Args:
            k_logits:
            topk_probs:

        Returns:

        """
        ensemble_logits = paddle.einsum('ijk,ij->ik', k_logits, topk_probs)
        return ensemble_logits


# ===================== Base Classes =====================
class BaseTrainer(object):
    """Base trainer
    """

    def __init__(self,
                 args,
                 tokenizer,
                 model,
                 pretrained_model=None,
                 indicator=None,
                 ensemble=None,
                 train_dataset=None,
                 eval_dataset=None,
                 data_collator=None,
                 loss_fn=None,
                 optimizer=None,
                 compute_metrics=None,
                 visual_writer=None):
        """init function

        Args:
            args: training args
            tokenizer: convert sequence to ids
            model: downstream task model
            pretrained_model: pretrained model
            indicator: indicator classifier
            ensemble: ensemble model
            train_dataset: dataset for training
            eval_dataset: dataset for evaluation
            data_collator: data collator
            loss_fn: loss function
            optimizer: optimizer for training
            compute_metrics: metrics function
            visual_writer: visualdl writer

        Returns:
            None
        """
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.pretrained_model = pretrained_model
        self.indicator = indicator
        self.ensemble = ensemble
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.compute_metrics = compute_metrics
        # default name_pbar is the first metric
        self.name_pbar = self.compute_metrics.metrics[0]
        self.visual_writer = visual_writer
        self.max_metric = 0.
        self.max_model_dir = ""
        # init dataloaders
        self._prepare_dataloaders()

    def _get_dataloader(self, dataset, sampler):
        """get dataloader, private function
        Args:
            dataset: dataset to be loaded
            sampler: data sampler

        Returns:
            paddle.io.Dataloader: data loader
        """
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )

    def _get_sampler(self, dataset):
        """get data sampler, private function
        Args:
            dataset: dataset to be loaded

        Returns:
            paddle.io.BatchSampler: data sampler
        """
        return BatchSampler(dataset=dataset,
                            shuffle=True,
                            batch_size=self.args.batch_size,
                            drop_last=self.args.dataloader_drop_last)

    def _prepare_dataloaders(self):
        """prepare dataloaders, private function
        Returns:
            None
        """
        if self.train_dataset:
            train_sampler = self._get_sampler(self.train_dataset)
            self.train_dataloader = self._get_dataloader(
                self.train_dataset, train_sampler)

        if self.eval_dataset:
            eval_sampler = self._get_sampler(self.eval_dataset)
            self.eval_dataloader = self._get_dataloader(
                self.eval_dataset, eval_sampler)

    def save_model(self, metrics_dataset, epoch):
        """
        Save model after epoch training in save_dir.
        Args:
            metrics_dataset: metrics of dataset
            epoch: training epoch number

        Returns:
            None
        """
        if metrics_dataset[self.name_pbar] > self.max_metric:
            self.max_metric = metrics_dataset[self.name_pbar]
            if os.path.exists(self.max_model_dir):
                print("Remove old max model dir:", self.max_model_dir)
                shutil.rmtree(self.max_model_dir)

            self.max_model_dir = osp.join(
                self.args.output, "epoch_" + str(epoch))
            os.makedirs(self.max_model_dir)
            save_model_path = osp.join(
                self.max_model_dir, "model_state.pdparams")
            paddle.save(self.model.state_dict(), save_model_path)
            print("Model saved at:", save_model_path)

    def get_input_ids_ind_topk(self, input_ids, seq_lens, indicator):
        """Get top k input ids with indications and return them.

        Args:
            input_ids: (B, L)
            seq_lens: (B) or (B, Ch)
            indicator:

        Returns:
            (token ids, probabilities)
        """
        input_ids = paddle.unsqueeze(input_ids, axis=0)
        (B, max_seq_len) = input_ids.shape

        token_to_idx = self.tokenizer.vocab.token_to_idx
        ind_id = token_to_idx[self.tokenizer.ind_token]
        sep_id = token_to_idx[self.tokenizer.sep_token]

        is_cross = (seq_lens <= max_seq_len - 2)  # remove [SEP] & [LABEL] token
        ind_positions = paddle.where(
            is_cross, x=seq_lens - 1, y=max_seq_len - 3)
        label_positions = ind_positions + 1
        sep_positions = ind_positions + 2
        for b in range(B):
            input_ids[b, ind_positions[b]] = ind_id
            input_ids[b, sep_positions[b]] = sep_id

        masked_positions = copy.deepcopy(label_positions)
        for i in range(1, B):
            masked_positions[i] += i * max_seq_len
        with paddle.no_grad():
            ind_logtis = indicator(input_ids=input_ids,
                                   masked_positions=masked_positions)
            ind_logits = ind_logtis.detach()  # [B, vocab_size]

        # remove special tokens and bases (A, T, C, G)
        ind_logits[:, :7] = float('-inf')
        ind_logits[:, 35:] = float('-inf')
        ind_probs = F.softmax(ind_logits, axis=-1)
        # (B, k), (B)
        topk_probs, topk_ind_ids = paddle.topk(
            ind_probs, self.args.top_k, axis=-1, largest=True)
        # (B*Ch, k, max_seq_len)
        input_ids_inds = paddle.tile(input_ids.unsqueeze(
            axis=1), repeat_times=(1, self.args.top_k, 1))

        for b in range(B):
            input_ids_inds[b, :, label_positions[b]] = paddle.t(topk_ind_ids[b])
        # tested, add it has more accuracy
        topk_probs = topk_probs / paddle.sum(topk_probs, axis=1, keepdim=True)

        return input_ids_inds, topk_probs, topk_ind_ids

    def train(self, epoch):
        """
        Args:
            epoch: training epoch

        Returns:
            None
        """
        raise NotImplementedError("Must implement train method.")

    def eval(self, epoch):
        """
        Args:
            epoch: eval epoch

        Returns:
            None
        """
        raise NotImplementedError("Must implement eval method.")


class BaseCollator(object):
    """Data collator that will dynamically pad the inputs to the longest sequence in the batch and process them.
    """

    def __init__(self):
        """
        """
        self.stack_fn = Stack()

    def __call__(self, data):
        """
        Args:
            data: instance for stack

        Returns:
            dict for trainer
        """
        raise NotImplementedError("Must implement __call__ method.")


class BaseInstance(object):
    """A single instance for data collator.
    """

    def __init__(self):
        """
        """
        pass

    def __call__(self):
        """
        Returns:
            class properties
        """
        return vars(self).items()


class BaseMetrics(abc.ABC):
    """Base class for functional tasks metrics
    """

    def __init__(self, metrics):
        """
        Args:
            metrics: names in list
        """
        self.metrics = [x.lower() for x in metrics]

    @abc.abstractmethod
    def __call__(self, outputs, labels):
        """
        Args:
            kwargs: required args of model (dict)

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
                if func == self.auc:
                    # given two neural outputs, calculate their logits
                    # and then calculate auc
                    logits = F.sigmoid(outputs).cpu().numpy()
                    m = func(logits, labels)
                else:
                    m = func(preds, labels)
                res[name] = m
            else:
                raise NotImplementedError
        return res

    @staticmethod
    def accuracy(preds, labels):
        """
        All args have same shapes.
        Args:
            preds: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            accuracy
        """
        return accuracy_score(labels, preds)

    @staticmethod
    def precision(preds, labels):
        """
        All args have same shapes.
        Args:
            preds: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            precision
        """
        return precision_score(labels, preds, average='macro')

    @staticmethod
    def recall(preds, labels):
        """
        All args have same shapes.
        Args:
            preds: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            precision
        """
        return recall_score(labels, preds, average='macro')

    @staticmethod
    def f1s(preds, labels):
        """
        All args have same shapes.
        Args:
            preds: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            precision
        """
        return f1_score(labels, preds, average='macro')

    @staticmethod
    def mcc(preds, labels):
        """
        All args have same shapes.
        Args:
            preds: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            precision
        """
        return matthews_corrcoef(labels, preds)

    @staticmethod
    def auc(preds, labels):
        """
        All args have same shapes.
        Args:
            preds: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            precision
        """
        labels += 1
        preds = preds[:, 1]
        return roc_auc_score(labels, preds)


class BaseStructMetrics(abc.ABC):
    """Base class for evaluation metrics
    """

    def __init__(self, metrics):
        """
        Args:
            metrics: names in list
        """
        self.metrics = [x.lower() for x in metrics]

    @abc.abstractmethod
    def __call__(self, **kwargs):
        """
        Args:
            kwargs: required args of model (dict)

        Returns:
            metrics in dict
        """
        raise NotImplementedError("Must implement __call__ method.")

    @staticmethod
    def accuracy(logits, labels, mask=None):
        """
        All args have same shapes.
        Args:
            logits: predictions of model, (batch_size, *)
            labels: ground truth, (batch_size, *)
            mask: valid sequence length masking, (batch_size, *)

        Returns:
            accuracy
        """
        if mask is not None:
            tp = np.sum(np.logical_and(np.logical_and(
                np.equal(labels, 1), np.equal(logits, 1)), mask))
            tn = np.sum(np.logical_and(np.logical_and(
                np.equal(labels, 0), np.equal(logits, 0)), mask))
            return (tp + tn) / np.sum(np.equal(mask, 1))
        else:
            tp = np.sum(np.logical_and(
                np.equal(labels, 1), np.equal(logits, 1)))
            tn = np.sum(np.logical_and(
                np.equal(labels, 0), np.equal(logits, 0)), )
            return (tp + tn) / labels.shape[0]

    @staticmethod
    def precision(logits, labels, mask=None):
        """
        All args have same shapes.
        Args:
            logits: predictions of model, (batch_size, *)
            labels: ground truth, (batch_size, *)
            mask: valid sequence length masking, (batch_size, *)

        Returns:
            precision
        """
        if mask is not None:
            tp = np.sum(np.logical_and(np.logical_and(
                np.equal(labels, 1), np.equal(logits, 1)), mask))
            fp = np.sum(np.logical_and(np.logical_and(
                np.equal(labels, 0), np.equal(logits, 1)), mask))
        else:
            tp = np.sum(np.logical_and(
                np.equal(labels, 1), np.equal(logits, 1)))
            fp = np.sum(np.logical_and(
                np.equal(labels, 0), np.equal(logits, 1)), )
        return tp / (tp + fp)

    @staticmethod
    def recall(logits, labels, mask=None):
        """
        All args have same shapes.
        Args:
            logits: predictions of model, (batch_size, *)
            labels: ground truth, (batch_size, *)
            mask: valid sequence length masking, (batch_size, *)

        Returns:
            recall
        """
        if mask is not None:
            tp = np.sum(np.logical_and(np.logical_and(
                np.equal(labels, 1), np.equal(logits, 1)), mask))
            fn = np.sum(np.logical_and(np.logical_and(
                np.equal(labels, 1), np.equal(logits, 0)), mask))
        else:
            tp = np.sum(np.logical_and(
                np.equal(labels, 1), np.equal(logits, 1)))
            fn = np.sum(np.logical_and(
                np.equal(labels, 1), np.equal(logits, 0)), )
        return tp / (tp + fn)

    def f1s(self, logits, labels, mask=None):
        """
        All args have same shapes.
        Args:
            logits: predictions of model, (batch_size, *)
            labels: ground truth, (batch_size, *)
            mask: valid sequence length masking, (batch_size, *)

        Returns:
            f1s
        """
        p = self.precision(logits, labels, mask)
        r = self.recall(logits, labels, mask)
        metric = 2 * p * r / (p + r)
        return metric


class FuncMetrics(object):
    """Metrics for classification
    """

    def __init__(self, metrics):
        """
        Args:
            metrics: names in string
        """
        self.metrics = [x.lower() for x in metrics]

    def __call__(self, outputs, labels):
        """
        Args:
            outputs: output of model
            labels: ground truth of data

        Returns:
            metrics in dict
        """
        preds = paddle.concat(outputs, axis=0)
        labels = paddle.concat(labels, axis=0)

        res = {}
        for name in self.metrics:
            func = getattr(self, name)
            if func:
                m = func(preds, labels)
                res[name] = m
            else:
                raise NotImplementedError(
                    "Metric {} is not implemented.".format(name))
        return res
