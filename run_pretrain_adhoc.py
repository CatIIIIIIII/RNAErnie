"""
This module runs RNAErnie pretrain with ad-hoc loss.

Author: wangning(wangning.roci@gmail.com)
Date  : 2024/1/26 1:21 PM
"""
# built-in modules
import argparse
import os
import os.path as osp
from functools import partial
# 3rd-party modules
from ahocorapy.keywordtree import KeywordTree
# paddle modules
import paddle
from paddlenlp.transformers import (
    ErnieModel,
    ErnieForMaskedLM,
    ErniePretrainingCriterion,
    LinearAnnealingWithWarmupDecay,
)
from paddlenlp.trainer import PdArgumentParser, get_last_checkpoint
from paddlenlp.datasets import MapDataset
from paddlenlp.utils.log import logger
# self-defined modules
from arg_utils import (
    str2bool,
    str2list,
    set_seed,
    default_logdir,
    list2str
)
from tokenizer_nuc import NUCTokenizer
from rna_pretrainer import (
    ErniePretrainingCriterion_adhoc,
    ErnieForMaskedLM_adhoc,
    load_motif,
    PretrainingTrainer,
    CriterionWrapper,
    PreTrainingArguments,
    PreFastaDataset,
    PreDataCollator,
    convert_text_to_pretrain
)

# ========== Define constants
STRATEGIES = ["BERT", "ERNIE", "MOTIF", "PROMPT", "ADHOC"]
RNA_LABELS = ['RNase_MRP_RNA',
              'RNase_P_RNA',
              'SRP_RNA',
              'Y_RNA',
              'antisense_RNA',
              'autocatalytically_spliced_intron',
              'guide_RNA',
              'hammerhead_ribozyme',
              'lncRNA',
              'miRNA',
              'misc_RNA',
              'ncRNA',
              'other',
              'piRNA',
              'pre_miRNA',
              'precursor_RNA',
              'rRNA',
              'ribozyme',
              'sRNA',
              'scRNA',
              'scaRNA',
              'siRNA',
              'snRNA',
              'snoRNA',
              'tRNA',
              'telomerase_RNA',
              'tmRNA',
              'vault_RNA']

# ========== Configuration
parser = argparse.ArgumentParser('Implementation of RNA pretrain.')
# model args
parser.add_argument('--model_name_or_path', type=str, default="ernie-1.0",
                    help='The build-in pretrained LM or the path to local model parameters.')
parser.add_argument('--model_type', type=str, default="ernie",
                    help='The build-in large model type.')
parser.add_argument('--max_seq_length', type=int, default=512,
                    help='The maximum length for model input, including special tokens.')

# data args
# parser.add_argument('--dataset', type=str, default="lnRC", choices=DATASETS, help='The dataset name to use.')
parser.add_argument('--dataset_dir', type=str,
                    default="./data/pre_random", help='Local path for dataset.')
parser.add_argument('--dataset_name', type=str,
                    default="group", help='Local path for dataset.')
parser.add_argument('--k_mer', type=int, default=1,
                    help='Number of continuous nucleic acids to form a token.')
parser.add_argument('--vocab_path', type=str,
                    default="./data/vocab/", help='Local path for vocab file.')
parser.add_argument('--vocab_size', type=int, default=39,
                    help='The size of vocobulary.')
parser.add_argument('--masked_lm_prob', type=float,
                    default=0.15, help='Mask token probability.')
parser.add_argument('--pre_strategy', type=str2list,
                    default="BERT,ERNIE,MOTIF,ADHOC",
                    help='Masking strategies, can accept concatenated ones with ","')
parser.add_argument('--num_consumed_samples', type=int,
                    default=0, help='Total consumed pretrain examples number.')
parser.add_argument('--num_samples_per_file', type=int,
                    default=300000, help='Examples number in 1 fasta group file.')
parser.add_argument('--num_groups', type=int, default=86,
                    help='Total pretrain group file number.')
parser.add_argument('--motif_dir', type=str,
                    default="./data/motif", help='Motif file root directory.')
parser.add_argument('--motif_files', type=str,
                    default="ATtRACT,SpliceAid,Statistics", help='Motif files to use.')

# training args
parser.add_argument('--device', type=str, default='gpu', choices=['gpu', 'cpu'],
                    help='Device for training, default to gpu.')
parser.add_argument('--seed', type=int, default=1000, help='Random seed.')
parser.add_argument('--disable_tqdm', type=str2bool, default=False,
                    help='Disable tqdm display if true.')
parser.add_argument('--do_train', type=str2bool, default=True,
                    help='Whether train model.')

parser.add_argument('--output_dir', type=str, default="./output",
                    help='The output directory where the model predictions and checkpoints will be written. ')
parser.add_argument('--overwrite_output_dir', type=str2bool, default=True,
                    help='Overwrite the content of the output directory. Use this to continue training if output_dir '
                         'points to a checkpoint directory. ')

parser.add_argument('--dataloader_num_workers', type=int,
                    default=50, help='The number of threads used by dataloader.')
parser.add_argument('--batch_size', type=int, default=50,
                    help='The number of samples used per step & per device.')
parser.add_argument('--per_device_train_batch_size', type=int, default=0,
                    help='The number of samples used for train, default same to batch_size.')
parser.add_argument('--per_device_eval_batch_size', type=int, default=0,
                    help='The number of samples used for eval, default same to batch_size.')

parser.add_argument('--num_train_epochs', type=int, default=1,
                    help='The number of epoch for training.')
parser.add_argument('--max_steps', type=int, default=-1,
                    help='Maximum steps, which overwrites num_epoch.')

parser.add_argument('--warmup_ratio', type=float, default=0.02,
                    help='Linear warmup over warmup_ratio fraction of total steps.')
parser.add_argument('--decay_steps', type=int, default=-1,
                    help='The steps for learning rate decay, default to total steps.')
parser.add_argument('--min_learning_rate', type=float,
                    default=1e-4, help='Minimal learning rate after decay.')
parser.add_argument('--learning_rate', type=float, default=1e-4,
                    help='The learning rate of optimizer.')

parser.add_argument('--save_steps', type=int, default=10000,
                    help='Save model every save_steps steps.')
parser.add_argument('--save_total_limit', type=int,
                    default=50, help='Maximum saving times.')
parser.add_argument('--logging_steps', type=int, default=1000,
                    help='Print logs every logging_step steps.')
parser.add_argument('--report_to', type=list, default=[
                    "visualdl"], help='The list of integrations to report the results and logs to.')

parser.add_argument('--max_grad_norm', type=float, default=1.0,
                    help='The maximum norm of all parameters.')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='Weight decay if we apply some.')
args = parser.parse_args()

# post process
if ".txt" not in args.vocab_path:
    args.vocab_path = osp.join(args.vocab_path,
                               "vocab_" + str(args.k_mer) + "MER.txt")  # expected: "./data/vocab/vocab_6MER.txt"
# "PROMPT" and "ADHOC" can't be used together
if "PROMPT" in args.pre_strategy and "ADHOC" in args.pre_strategy:
    raise ValueError("PROMPT and ADHOC can't be used together!")
args.output_dir = osp.join(args.output_dir, list2str(
    args.pre_strategy))  # expected: "./output/bert,ernie"
# expected: "./output/bert,ernie/runs/Oct08_21-19-09_suzhou-5"
args.logging_dir = osp.join(args.output_dir, default_logdir())

args.per_device_train_batch_size = args.batch_size
args.per_device_eval_batch_size = args.batch_size

logger.debug("Loading configuration.")
PdAparser = PdArgumentParser((PreTrainingArguments))
training_args, = PdAparser.parse_args_into_dataclasses()

# if config.dataset in DATASETS:
# if you custom you hyper-parameters in parser, it will overwrite all args.
logger.debug("Over-writing arguments by user config!")
for arg in vars(training_args):
    if arg in vars(args):
        setattr(training_args, arg, getattr(args, arg))

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

# ========== Log on each process the small summary:
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
    +
    f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)

# ========== Detecting last checkpoint.
logger.debug("Detecting last checkpoint.")
last_checkpoint = None
if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. "
            f"To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )

# ========== Build tokenizer, model, criterion
logger.debug("Building tokenizer, model, criterion.")
# load tokenizer
logger.debug("Loading tokenization.")
tokenizer = NUCTokenizer(args.k_mer, args.vocab_path)
assert tokenizer.vocab_size == args.vocab_size, "Vocab size of tokenizer must be equal to args.vocab_size."

# load base model, pretrain model
if "ADHOC" not in args.pre_strategy:
    base_class, model_class, criterion_class = ErnieModel, ErnieForMaskedLM, ErniePretrainingCriterion
else:
    base_class, model_class, criterion_class = ErnieModel, ErnieForMaskedLM_adhoc, ErniePretrainingCriterion_adhoc
pretrained_models_list = list(model_class.pretrained_init_configuration.keys())
if args.model_name_or_path in pretrained_models_list:
    model_config = model_class.pretrained_init_configuration[args.model_name_or_path]
    model_config["vocab_size"] = tokenizer.vocab_size
    model = model_class(base_class(**model_config))
else:
    model = model_class.from_pretrained(
        args.model_name_or_path,
    )
# load criterion
criterion = criterion_class()
# log model config
model_config = model.get_model_config()["init_args"][0]
for k, v in model_config.items():
    setattr(args, k, v)

# ========== Prepare data
# load motifs
logger.debug("Preparing data.")
motif_tree_dict = None
if "MOTIF" in args.pre_strategy:
    motif_dict = load_motif(motif_dir=args.motif_dir,
                            motif_name=args.motif_files,
                            tokenizer=tokenizer)

    motif_tree_dict = {}
    motif_tree = KeywordTree()
    for k, v in motif_dict.items():
        if k != "Statistics":
            for m in v:
                motif_tree.add(m)
    motif_tree.finalize()
    motif_tree_dict["DataBases"] = motif_tree

    motif_tree = KeywordTree()
    for k, v in motif_dict.items():
        if k == "Statistics":
            for m in v:
                motif_tree.add(m)
    motif_tree.finalize()
    motif_tree_dict["Statistics"] = motif_tree

# load datasets
trans_func = partial(
    convert_text_to_pretrain,
    tokenizer=tokenizer,
    pre_strategy=args.pre_strategy,
    max_seq_length=args.max_seq_length,
    masked_lm_prob=args.masked_lm_prob,
    motif_tree_dict=motif_tree_dict,
    seed=training_args.seed,
)
raw_dataset = PreFastaDataset(
    fasta_dir=args.dataset_dir,
    prefix=args.dataset_name,
    num_file=args.num_groups,
    num_samples_per_file=args.num_samples_per_file,
    tokenizer=tokenizer,
    num_specials=4 if "PROMPT" in args.pre_strategy else 2
)
m_dataset = MapDataset(raw_dataset)
m_dataset.map(trans_func)

# ========== Create the learning_rate scheduler and optimizer
logger.debug("Creating learning rate scheduler and optimizer.")
# scheduler
if training_args.max_steps == -1:
    total_train_batch_size = training_args.per_device_train_batch_size * \
        training_args.gradient_accumulation_steps * \
        training_args.world_size
    training_args.max_steps = len(
        m_dataset) // int(total_train_batch_size) * training_args.num_train_epochs
if training_args.decay_steps == -1:
    training_args.decay_steps = training_args.max_steps
warmup_steps = int(training_args.warmup_ratio * training_args.max_steps)
lr_scheduler = LinearAnnealingWithWarmupDecay(
    training_args.learning_rate,
    training_args.min_learning_rate,
    warmup_step=warmup_steps,
    decay_step=training_args.decay_steps
)

# ========== Training
logger.debug("Start training.")
_collate_fn = PreDataCollator(tokenizer.pad_token_id)
# train model
trainer = PretrainingTrainer(
    model=model,
    criterion=CriterionWrapper(criterion),
    args=training_args,
    data_collator=_collate_fn,
    train_dataset=m_dataset,
    optimizers=(None, lr_scheduler),
    tokenizer=tokenizer,
)
# checkpoints
checkpoint = None
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
elif last_checkpoint is not None:
    checkpoint = last_checkpoint
# start
train_result = trainer.train(resume_from_checkpoint=checkpoint)
metrics = train_result.metrics
trainer.save_model()
