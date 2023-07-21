"""
This module runs secondary structure prediction.

Author: wangning(wangning.roci@gmail.com)
Date  : 2022/11/16 9:08
"""

import argparse
import os.path as osp
from functools import partial

import paddle
from paddlenlp.datasets import MapDataset
from paddlenlp.utils.log import logger
from paddlenlp.transformers import ErnieModel

from arg_utils import (
    default_logdir,
    print_config,
    set_seed,
    str2bool,
    str2list
)
from base_classes import IndicatorClassifier
from tokenizer_nuc import NUCTokenizer
import param_turner2004
from secondary_structure import (
    BPseqDataset,
    convert_instance_to_ssp,
    MixedFold,
    StructuredLoss,
    SspCollator,
    SspMetric,
    SspTrainer
)
from visualizer import Visualizer


# ========== Define constants
TASKS = ["RNAStrAlign", "bpRNA1m"]
DATASETS = {
    "RNAStrAlign": ("RNAStrAlign600.lst", "archiveII600.lst"),
    "bpRNA1m": ("TR0.lst", "TS0.lst"),
}
# ========== Configuration
logger.debug("Loading configuration.")
parser = argparse.ArgumentParser(
    description='RNA secondary structure prediction using deep learning with thermodynamic integrations', add_help=True)
# model args
parser.add_argument('--model_name_or_path',
                    type=str,
                    default="./output/BERT,ERNIE,MOTIF,PROMPT/checkpoint-final",
                    help='The build-in pretrained LM or the path to local model parameters.')
parser.add_argument('--model_path',
                    type=str,
                    default="./output_ft/ssp/RNAStrAlign/BERT,ERNIE,MOTIF,PROMPT",
                    help='The build-in pretrained LM or the path to local model parameters.')
parser.add_argument('--two_stage', type=str2bool, default=False, help='Whether use two stage.')
parser.add_argument('--top_k', type=int, default=3, help='Select top k indications for stage two.')
parser.add_argument('--with_pretrain', type=str2bool, default=True)

# data args
parser.add_argument('--task_name', type=str, default="RNAStrAlign", choices=TASKS, help='Task name of training data.')
parser.add_argument('--k_mer', type=int, default=1, help='Number of continuous nucleic acids to form a token.')
parser.add_argument('--vocab_path', type=str, default="./data/vocab/", help='Local path for vocab file.')
parser.add_argument('--vocab_size', type=int, default=39, help='The size of vocobulary.')
parser.add_argument('--dataloader_num_workers', type=int, default=8, help='The number of threads used by dataloader.')
parser.add_argument('--dataloader_drop_last', type=str2bool, default=True, help='Whether drop the last batch sample.')
parser.add_argument('--dataset_dir', type=str, default="./data/ft/ssp", help='Local path for dataset.')

# training args
parser.add_argument('--train', type=str2bool, default=True, help='Whether train the model.')
parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--disable_tqdm', type=str2bool, default=False, help='Disable tqdm display if true.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--device',
                    type=str,
                    default='gpu',
                    choices=['gpu', 'cpu'],
                    help='Device for training, default to gpu.')

parser.add_argument('--num_train_epochs', type=int, default=50, help='The number of epoch for training.')
parser.add_argument('--batch_size', type=int, default=1, help='The number of samples used per step, must be 1.')
parser.add_argument('--lr', type=float, default=0.001, help='the learning rate for optimizer (default: 0.001)')
parser.add_argument('--l1-weight', type=float, default=0., help='the weight for L1 regularization (default: 0)')
parser.add_argument('--l2-weight', type=float, default=0., help='the weight for L2 regularization (default: 0)')
parser.add_argument('--loss_pos_paired',
                    type=float,
                    default=0.5,
                    help='the penalty for positive base-pairs for loss augmentation (default: 0.5)')
parser.add_argument('--loss_neg_paired',
                    type=float,
                    default=0.005,
                    help='the penalty for negative base-pairs for loss augmentation (default: 0.005)')
parser.add_argument('--loss_pos_unpaired',
                    type=float,
                    default=0,
                    help='the penalty for positive unpaired bases for loss augmentation (default: 0)')
parser.add_argument('--loss_neg_unpaired',
                    type=float,
                    default=0,
                    help='the penalty for negative unpaired bases for loss augmentation (default: 0)')
parser.add_argument('--metrics',
                    type=str2list,
                    default="F1s,Accuracy,Precision,Recall",
                    help='Use which metrics to evaluate model, could be concatenate by ",".')

# logging args
parser.add_argument('--output', type=str, default="./output_ft/ssp", help='Logging directory.')
parser.add_argument('--visualdl_dir', type=str, default="visualdl", help='Visualdl logging directory.')
parser.add_argument('--logging_steps', type=int, default=100, help='Update visualdl logs every logging_steps.')
parser.add_argument('--save_max', type=str2bool, default=True, help='Save model with max metric.')

args = parser.parse_args()


if __name__ == "__main__":
    # ========== post process
    if ".txt" not in args.vocab_path:
        # expected: "./data/vocab/vocab_6MER.txt"
        args.vocab_path = osp.join(args.vocab_path, "vocab_" + str(args.k_mer) + "MER.txt")
    ct = default_logdir()
    args.output = osp.join(osp.join(args.output, args.task_name), ct)
    args.dataset_train = osp.join(args.dataset_dir, DATASETS[args.task_name][0])
    args.dataset_test = osp.join(args.dataset_dir, DATASETS[args.task_name][1])
    if args.model_path.split(".") != "pdparams":
        args.model_path = osp.join(args.model_path, "model_state.pdparams")
    args.visualdl_dir = osp.join(args.output, args.visualdl_dir)
    print_config(args, "RNA Secondary Structure Prediction.")

    # ========== Set random seeds
    logger.debug("Set random seeds.")
    set_seed(args.seed)

    # ========== Set device
    logger.debug("Set device.")
    paddle.set_device(args.device)

    # ========== Set multi-gpus environment
    logger.debug("Set multi-gpus environment.")
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

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
    pretrained_model = ErnieModel.from_pretrained(args.model_name_or_path)
    # load model
    config = {
        'max_helix_length': 30,
        'embed_size': 64,
        'num_filters': (64, 64, 64, 64, 64, 64, 64, 64),
        'filter_size': (5, 3, 5, 3, 5, 3, 5, 3),
        'pool_size': (1, ),
        'dilation': 0,
        'num_lstm_layers': 2,
        'num_lstm_units': 32,
        'num_transformer_layers': 0,
        'num_hidden_units': (32, ),
        'num_paired_filters': (64, 64, 64, 64, 64, 64, 64, 64),
        'paired_filter_size': (5, 3, 5, 3, 5, 3, 5, 3),
        'dropout_rate': 0.5,
        'fc_dropout_rate': 0.5,
        'num_att': 8,
        'pair_join': 'cat',
        'no_split_lr': False,
        'n_out_paired_layers': 3,
        'n_out_unpaired_layers': 0,
        'exclude_diag': True
    }
    model = MixedFold(init_param=param_turner2004, **config)
    if not args.train:
        model.set_state_dict(paddle.load(args.model_path))
    indicator = None if not args.two_stage else IndicatorClassifier(model_name_or_path=args.model_name_or_path)
    # load loss function
    _loss_fn = StructuredLoss(loss_pos_paired=args.loss_pos_paired,
                              loss_neg_paired=args.loss_neg_paired,
                              loss_pos_unpaired=args.loss_pos_unpaired,
                              loss_neg_unpaired=args.loss_neg_unpaired,
                              l1_weight=args.l1_weight,
                              l2_weight=args.l2_weight)

    # ========== Prepare data
    trans_func = partial(
        convert_instance_to_ssp,
        tokenizer=tokenizer,
    )
    train_dataset = BPseqDataset(args.dataset_dir, args.dataset_train)
    m_train_dataset = MapDataset(train_dataset)
    m_train_dataset.map(trans_func)
    test_dataset = BPseqDataset(args.dataset_dir, args.dataset_test)
    m_test_dataset = MapDataset(test_dataset)
    m_test_dataset.map(trans_func)

    # ========== Create the learning_rate scheduler (if need) and optimizer
    _optimizer = paddle.optimizer.Adam(parameters=model.parameters())

    # ========== Create visualizer
    if args.train:
        _visualizer = Visualizer(log_dir=args.visualdl_dir,
                                 name="Secondary Structure Prediction, " + args.task_name + ", " + ct)
    else:
        _visualizer = None

    # ========== Training
    _collate_fn = SspCollator()
    _metric = SspMetric(args.metrics)
    ssp_trainer = SspTrainer(
        args=args,
        tokenizer=tokenizer,
        model=model,
        pretrained_model=pretrained_model,
        indicator=indicator,
        train_dataset=m_train_dataset,
        eval_dataset=m_test_dataset,
        data_collator=_collate_fn,
        loss_fn=_loss_fn,
        optimizer=_optimizer,
        compute_metrics=_metric,
        visual_writer=_visualizer,
    )
    if args.train:
        for i_epoch in range(args.num_train_epochs):
            if not ssp_trainer.get_status():
                print("Epoch: {}".format(i_epoch))
                ssp_trainer.train(i_epoch)
                ssp_trainer.eval(i_epoch)
    else:
        ssp_trainer.eval(0)
