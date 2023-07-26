"""
This module runs rna rna interaction prediction.

Author: wangning(wangning.roci@gmail.com)
Date  : 2022/12/7 7:41 PM
"""

import os.path as osp
import argparse
from functools import partial

import paddle
from paddlenlp.utils.log import logger
from paddlenlp.transformers import ErnieModel
from paddlenlp.datasets import MapDataset

from arg_utils import (
    default_logdir,
    str2bool,
    set_seed,
    print_config,
    str2list
)
from rna_rna_interaction import (
    convert_instance_to_rr,
    RRInterMetrics,
    RRDataCollator,
    RRInterCriterionWrapper,
    ErnieRRInter,
    RRInterTrainer,
    GenerateRRInterTrainTest
)
from tokenizer_nuc import NUCTokenizer
from visualizer import Visualizer

# ========== Define constants
DATASETS = ["lncRNASNP2", "MirTarRAW"]

MAX_SEQ_LEN = {
    "lncRNASNP2": 2048,
    "MirTarRAW": [26, 40],
}
# ========== Configuration
logger.info("Loading configuration.")
parser = argparse.ArgumentParser('Implementation of RNA-RNA Interaction prediction.')
# model args
parser.add_argument('--model_name_or_path', type=str, default="./output/BERT,ERNIE,MOTIF,PROMPT/checkpoint-final/",
                    help='The pretrain model for feature extraction.')
parser.add_argument('--with_pretrain', type=str2bool, default=True, help='Whether use original channels.')
parser.add_argument('--proj_size', type=int, default=64, help='Project pretrained features to this size.')
parser.add_argument('--model_path',
                    type=str,
                    default="./output_ft/rr_inter/MirTarRAW/BERT,ERNIE,MOTIF,PROMPT/model_state.pdparams",
                    help='The build-in pretrained LM or the path to local model parameters.')

# data args
parser.add_argument('--dataset', type=str, default="MirTarRAW", choices=DATASETS,
                    help='The file list to train.')
parser.add_argument('--dataset_dir', type=str, default="./data/ft/rr_inter/", help='Local path for dataset.')
parser.add_argument('--k_mer', type=int, default=1, help='Number of continuous nucleic acids to form a token.')
parser.add_argument('--vocab_path', type=str, default="./data/vocab/", help='Local path for vocab file.')
parser.add_argument('--dataloader_num_workers', type=int, default=16, help='The number of threads used by dataloader.')
parser.add_argument('--dataloader_drop_last', type=str2bool, default=True,
                    help='Whether drop the last sample.')
# training args
parser.add_argument('--device', type=str, default='gpu', choices=['gpu', 'cpu'],
                    help='Device for training, default to gpu.')
parser.add_argument('--seed', type=int, default=1000, help='Random seed.')
parser.add_argument('--disable_tqdm', type=str2bool, default=False,
                    help='Disable tqdm display if true.')
parser.add_argument('--fix_pretrain', type=str2bool, default=False,
                    help='Whether fix parameters of pretrained model.')

parser.add_argument('--train', type=str2bool, default=True, help='Whether train the model.')
parser.add_argument('--batch_size', type=int, default=256, help='The number of samples used per step & per device.')
parser.add_argument('--num_train_epochs', type=int, default=100, help='The number of epoch for training.')
parser.add_argument('--lr', type=float, default=1e-3, help='The learning rate of optimizer.')
parser.add_argument('--metrics', type=str2list, default="Accuracy,Recall,Precision,F1s",
                    help='Use which metrics to evaluate model, could be concatenate by ","'
                         'and the first one will show on pbar.')

# logging args
parser.add_argument('--logging_steps', type=int, default=100, help='Print logs every logging_step steps.')
parser.add_argument('--output', type=str, default="./output_ft/rr_inter", help='Logging directory.')
parser.add_argument('--visualdl_dir', type=str, default="visualdl", help='Visualdl logging directory.')
parser.add_argument('--save_max', type=str2bool, default=True, help='Save model with max metric.')

args = parser.parse_args()

if __name__ == "__main__":
    # ========== post process
    if ".txt" not in args.vocab_path:
        # expected: "./data/vocab/vocab_1MER.txt"
        args.vocab_path = osp.join(args.vocab_path, "vocab_" + str(args.k_mer) + "MER.txt")
    if args.model_path.split(".")[-1] != "pdparams":
        args.model_path = osp.join(args.model_path, "model_state.pdparams")
    ct = default_logdir()
    args.output = osp.join(osp.join(args.output, args.dataset), ct)
    args.visualdl_dir = osp.join(args.output, args.visualdl_dir)
    print_config(args, "RNA RNA Interaction")

    # ========== Set random seeds
    logger.info("Set random seeds.")
    set_seed(args.seed)

    # ========== Set device
    logger.info("Set device.")
    paddle.set_device(args.device)

    # ========== Build tokenizer, pretrained model, model, criterion
    logger.info("Build tokenizer, pretrained model, model, criterion.")
    # load tokenizer
    logger.info("Loading tokenization.")
    tokenizer_class = NUCTokenizer
    tokenizer = tokenizer_class(
        k_mer=args.k_mer,
        vocab_file=args.vocab_path
    )
    # load pretrained model
    logger.info("Loading pretrained model.")
    pretrained_model = None
    hidden_states_size = 768
    if args.with_pretrain:
        pretrained_model = ErnieModel.from_pretrained(args.model_name_or_path)
        model_config = pretrained_model.get_model_config()
        hidden_states_size = model_config["hidden_size"]  # for pretrained features extraction and projection
    model = ErnieRRInter(extractor=pretrained_model,
                         hidden_states_size=hidden_states_size,
                         proj_size=args.proj_size,
                         with_pretrain=args.with_pretrain,
                         fix_pretrain=args.fix_pretrain)
    if not args.train:
        model.set_state_dict(paddle.load(args.model_path))
    # model = ErnieForSequenceClassification.from_pretrained(args.model_name_or_path, num_classes=2)
    # load criterion
    _loss_fn = RRInterCriterionWrapper()

    # ========== Prepare data
    logger.info("Preparing data.")
    # load datasets
    # train & test datasets
    trans_func = partial(convert_instance_to_rr,
                         tokenizer=tokenizer,
                         max_seq_lens=MAX_SEQ_LEN[args.dataset])
    datasets_generator = GenerateRRInterTrainTest(rr_dir=args.dataset_dir,
                                                  dataset=args.dataset,
                                                  split=0.8,
                                                  seed=args.seed)
    raw_dataset_train, raw_dataset_test = datasets_generator.get()
    m_dataset_train, m_dataset_test = MapDataset(raw_dataset_train), MapDataset(raw_dataset_test)
    m_dataset_train.map(trans_func)
    m_dataset_test.map(trans_func)

    # ========== Create the learning_rate scheduler (if need) and optimizer
    logger.info("Creating learning rate scheduler and optimizer.")
    # optimizer
    optimizer = paddle.optimizer.AdamW(parameters=model.parameters(), learning_rate=args.lr)

    # ========== Create visualizer
    if args.train:
        _visualizer = Visualizer(log_dir=args.visualdl_dir, name="RNA RNA interaction, " + args.dataset + ", " + ct)
    else:
        _visualizer = None

    # ========== Training
    logger.info("Start training.")
    _collate_fn = RRDataCollator(max_seq_len=sum(MAX_SEQ_LEN[args.dataset]))
    _metric = RRInterMetrics(metrics=args.metrics)
    # train model
    rr_inter_trainer = RRInterTrainer(
        args=args,
        tokenizer=tokenizer,
        model=model,
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
            rr_inter_trainer.train(i_epoch)
            rr_inter_trainer.eval(i_epoch)
    else:
        # evaluate dataset only
        rr_inter_trainer.eval(0)
