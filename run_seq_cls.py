"""
This module runs sequence classification task.

Author: wangning(wangning.roci@gmail.com)
Date  : 2022/12/7 7:41 PM
"""

import argparse
import os.path as osp
from functools import partial

import paddle
from paddlenlp.datasets import MapDataset
from paddlenlp.transformers import ErnieForSequenceClassification
from paddlenlp.utils.log import logger

from arg_utils import (
    set_seed,
    str2bool,
    default_logdir,
    print_config,
    str2list
)
from tokenizer_nuc import NUCTokenizer
from base_classes import IndicatorClassifier, MajorityVoter
from sequence_classification import (
    SeqClsDataset,
    convert_instance_to_cls,
    SeqClsCollator,
    SeqClsMetrics,
    SeqClsLoss,
    SeqClsTrainer,
    LongSeqClsCollator,
    ErnieForLongSequenceClassification
)
from visualizer import Visualizer


# ========== Define constants
TASKS = ["nRC", "lncRNA_H", "lncRNA_M"]
NUM_CLASSES = {
    "nRC": 13,
    "lncRNA_H": 2,
    "lncRNA_M": 2,
}
LABEL2ID = {
    "nRC": {
        "5S_rRNA": 0,
        "5_8S_rRNA": 1,
        "tRNA": 2,
        "ribozyme": 3,
        "CD-box": 4,
        "Intron_gpI": 5,
        "Intron_gpII": 6,
        "riboswitch": 7,
        "IRES": 8,
        "HACA-box": 9,
        "scaRNA": 10,
        "leader": 11,
        "miRNA": 12
    },
    "lncRNA_H": {
        "lnc": 0,
        "pc": 1
    },
    "lncRNA_M": {
        "lnc": 0,
        "pc": 1
    },
}
MAX_MODEL_LEN = 512
MAX_CHUNK_NUMBER = 6
MAX_SEQ_LEN = {
    "nRC": MAX_MODEL_LEN,
    "lncRNA_H": MAX_MODEL_LEN * MAX_CHUNK_NUMBER,
    "lncRNA_M": MAX_MODEL_LEN * MAX_CHUNK_NUMBER,
}

# ========== Configuration
logger.debug("Loading configuration.")
parser = argparse.ArgumentParser('Implementation of RNA sequence classification.')
# model args
parser.add_argument('--model_name_or_path',
                    type=str,
                    default="./output/BERT,ERNIE,MOTIF,PROMPT/checkpoint_final",
                    help='The build-in pretrained LM or the path to local model parameters.')
parser.add_argument('--max_seq_len', type=int, default=0,
                    help='The maximum length for model input, including special tokens.')
parser.add_argument('--num_classes', type=int, default=0, help='The number of classes in task.')
parser.add_argument('--with_pretrain', type=str2bool, default=True, help='Whether use original channels.')
parser.add_argument('--hidden_states_size', type=int, default=768, help='The hidden size of model.')
parser.add_argument('--proj_size', type=int, default=128, help='Project pretrained features to this size.')
parser.add_argument('--model_path', type=str, default="./output_ft/seq_cls/nRC/BERT,ERNIE,MOTIF,PROMPT",
                    help='Pretrained down-stream task model weight.')
parser.add_argument('--two_stage', type=str2bool, default=False, help='Whether use two-stage pipeline.')
parser.add_argument('--top_k', type=int, default=3, help='Select top k indications for stage two.')

# data args
parser.add_argument('--k_mer', type=int, default=1, help='Number of continuous nucleic acids to form a token.')
parser.add_argument('--vocab_path', type=str, default="./data/vocab/", help='Local path for vocab file.')
parser.add_argument('--vocab_size', type=int, default=39, help='The size of vocobulary.')
parser.add_argument('--dataset', type=str, default="nRC", choices=TASKS, help='The dataset name to use.')
parser.add_argument('--dataset_dir', type=str, default="./data/ft/seq_cls", help='Local path for dataset.')
parser.add_argument('--dataloader_num_workers', type=int, default=8, help='The number of threads used by dataloader.')
parser.add_argument('--dataloader_drop_last', type=str2bool, default=True, help='Whether drop the last batch sample.')

# training args
parser.add_argument('--device', type=str, default='gpu', choices=['gpu', 'cpu'],
                    help='Device for training, default to gpu.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--fix_pretrain', type=str2bool, default=False, help='Whether fix parameters of pretrained model.')
parser.add_argument('--disable_tqdm', type=str2bool, default=False, help='Disable tqdm display if true.')
parser.add_argument('--max_chunk_number', type=int, default=MAX_CHUNK_NUMBER,
                    help='Maximum chunk number for long sequences.')
parser.add_argument('--use_chunk', type=str2bool, default=True,
                    help='Whether use chunk strategy when classify long sequences.')

parser.add_argument('--train', type=str2bool, default=True, help='Whether train the model.')
parser.add_argument('--batch_size', type=int, default=50, help='The number of samples used per step & per device.')
parser.add_argument('--num_train_epochs', type=int, default=50, help='The number of epoch for training.')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='The learning rate of optimizer.')
parser.add_argument('--metrics',
                    type=str2list,
                    default="F1s,Precision,Recall,Accuracy,Mcc",
                    help='Use which metrics to evaluate model, could be concatenate by ",".')

# logging args
parser.add_argument('--logging_steps', type=int, default=100, help='Update visualdl logs every logging_steps.')
parser.add_argument('--output', type=str, default="./output_ft/seq_cls", help='Output directory.')
parser.add_argument('--visualdl_dir', type=str, default="visualdl", help='Visualdl logging directory.')
parser.add_argument('--save_max', type=str2bool, default=True, help='Save model with max metric.')
args = parser.parse_args()

if __name__ == "__main__":
    # ========== post process
    if ".txt" not in args.vocab_path:
        args.vocab_path = osp.join(args.vocab_path,
                                   "vocab_" + str(args.k_mer) + "MER.txt")  # expected: "./data/vocab/vocab_6MER.txt"
    if args.model_path.split(".")[-1] != "pdparams":
        args.model_path = osp.join(args.model_path, "model_state.pdparams")
    ct = default_logdir()
    args.output = osp.join(osp.join(args.output, args.dataset), ct)
    args.num_classes = NUM_CLASSES[args.dataset]
    if args.max_seq_len == 0:
        if args.use_chunk:
            args.max_seq_len = MAX_SEQ_LEN[args.dataset]
        else:
            args.max_seq_len = MAX_MODEL_LEN
    args.visualdl_dir = osp.join(args.output, args.visualdl_dir)
    print_config(args, "RNA Sequence Classification")

    # ========== Set random seeds
    logger.debug("Set random seeds.")
    set_seed(args.seed)

    # ========== Set device
    logger.debug("Set device.")
    paddle.set_device(args.device)

    # ========== Build tokenizer, model, criterion
    logger.debug("Build tokenizer, model, criterion.")
    # load tokenizer
    logger.debug("Loading tokenization.")
    tokenizer = NUCTokenizer(
        k_mer=args.k_mer,
        vocab_file=args.vocab_path,
    )
    assert tokenizer.vocab_size == args.vocab_size, "Vocab size of tokenizer must be equal to args.vocab_size."
    # load pretrained model
    logger.info("Loading pretrained model.")
    if args.train:
        # train from scratch
        if args.dataset != "nRC" and args.use_chunk:
            model = ErnieForLongSequenceClassification.from_pretrained(args.model_name_or_path, num_classes=args.num_classes)
        else:
            model = ErnieForSequenceClassification.from_pretrained(args.model_name_or_path, num_classes=args.num_classes)
    else:
        if args.dataset != "nRC" and args.use_chunk:
            model = ErnieForLongSequenceClassification.from_pretrained(args.model_name_or_path, num_classes=args.num_classes)
            model.set_state_dict(paddle.load(args.model_path))
            
        else:
            model = ErnieForSequenceClassification.from_pretrained(args.model_name_or_path, num_classes=args.num_classes)
            model.set_state_dict(paddle.load(args.model_path))
            
    indicator = IndicatorClassifier(model_name_or_path=args.model_name_or_path) if args.two_stage else None
    ensemble = MajorityVoter() if args.two_stage else None
    # load loss function
    _loss_fn = SeqClsLoss()
    # load model
    logger.info("Loading model.")

    # ========== Prepare data
    logger.debug("Preparing data.")
    # load datasets
    trans_func = partial(convert_instance_to_cls, tokenizer=tokenizer, label2id=LABEL2ID[args.dataset])
    # train
    raw_dataset_train = SeqClsDataset(fasta_dir=args.dataset_dir, prefix=args.dataset, tokenizer=tokenizer)
    m_dataset_train = MapDataset(raw_dataset_train)
    m_dataset_train.map(trans_func)
    # test
    raw_dataset_test = SeqClsDataset(fasta_dir=args.dataset_dir, prefix=args.dataset, tokenizer=tokenizer, train=False)
    m_dataset_test = MapDataset(raw_dataset_test)
    m_dataset_test.map(trans_func)

    # ========== Create the learning_rate scheduler (if need) and optimizer
    logger.info("Creating learning rate scheduler and optimizer.")
    # optimizer
    optimizer = paddle.optimizer.AdamW(parameters=model.parameters(), learning_rate=args.learning_rate)

    # ========== Create visualizer
    if args.train:
        _visualizer = Visualizer(log_dir=args.visualdl_dir, name="Sequence classification, " + args.dataset + ", " + ct)
    else:
        _visualizer = None

    # ========== Training
    logger.debug("Start training.")
    if args.dataset != "nRC" and args.use_chunk:
        _collate_fn = LongSeqClsCollator(max_seq_len=args.max_seq_len, tokenizer=tokenizer)
    else:
        _collate_fn = SeqClsCollator(max_seq_len=args.max_seq_len, tokenizer=tokenizer)
    _metric = SeqClsMetrics(metrics=args.metrics)
    # train model
    seq_cls_trainer = SeqClsTrainer(
        args=args,
        tokenizer=tokenizer,
        model=model,
        indicator=indicator,
        ensemble=ensemble,
        train_dataset=m_dataset_train,
        eval_dataset=m_dataset_test,
        data_collator=_collate_fn,
        loss_fn=_loss_fn,
        optimizer=optimizer,
        compute_metrics=_metric,
        visual_writer=_visualizer
    )
    if args.train:
        for i_epoch in range(args.num_train_epochs):
            print("Epoch: {}".format(i_epoch))
            seq_cls_trainer.train(i_epoch)
            seq_cls_trainer.eval(i_epoch)
    else:
        # evaluate dataset only
        seq_cls_trainer.eval(0)
