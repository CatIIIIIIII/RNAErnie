[![DOI](https://zenodo.org/badge/665802008.svg)](https://zenodo.org/doi/10.5281/zenodo.10847620)

# RNAErnie

Official implement of paper "Multi-purpose RNA Language Modeling with Motif-aware Pre-training and Type-guided Fine-tuning" with [paddlepaddle](https://github.com/PaddlePaddle/Paddle/tree/develop).

This repository contains codes and pre-trained models for RNAErnie, which leverages RNA motifs as **biological priors** and proposes a **motif-level random masking** strategy to enhance pre-training tasks. Furthermore, RNAErnie improves sequence classfication, RNA-RNA interaction prediction, and RNA secondary structure prediction by fine-tuning or adapating on downstream tasks with two-stage **type-guided** learning. Our paper will be published soon.

![Overview](./images/overview.png)

- [RNAErnie](#rnaernie)
  - [Update Log](#update-log)
  - [Installation](#installation)
    - [Create Environment with Conda](#create-environment-with-conda)
    - [Run in Docker](#run-in-docker)
      - [Step1: Prepare code](#step1-prepare-code)
      - [Step2: Prepare running environment](#step2-prepare-running-environment)
      - [Step3: Run](#step3-run)
  - [Pre-training](#pre-training)
    - [1. Data Preparation](#1-data-preparation)
    - [2. Pre-training](#2-pre-training)
    - [3. Download Pre-trained Models](#3-download-pre-trained-models)
    - [4. Visualization](#4-visualization)
    - [5. Extract RNA Sequence Embeddings](#5-extract-rna-sequence-embeddings)
  - [Downstream Tasks](#downstream-tasks)
    - [RNA sequence classification](#rna-sequence-classification)
      - [1. Data Preparation](#1-data-preparation-1)
      - [2. Fine-tuning](#2-fine-tuning)
      - [3. Evaluation](#3-evaluation)
    - [RNA RNA interaction prediction](#rna-rna-interaction-prediction)
      - [1. Data Preparation](#1-data-preparation-2)
      - [2. Fine-tuning](#2-fine-tuning-1)
      - [3. Evaluation](#3-evaluation-1)
    - [RNA secondary structure prediction](#rna-secondary-structure-prediction)
      - [1. Data Preparation](#1-data-preparation-3)
      - [2. Adaptation](#2-adaptation)
      - [3. Evaluation](#3-evaluation-2)
  - [Baselines](#baselines)
  - [Citation](#citation)
  - [Future work](#future-work)


## Update Log

- 2024.06.26: Considering most of the researchers will prefer to use transformers and pytorch as backend. So, I transfer my work to transformers and train a pytorch model from scratch. The new model is trained with more powerful settings: The max model length is up to 2048 now and the pretraining dataset is the newest version of rnacentral, which contains about 31 million RNA sequences after length filtering (<2048). This pytorch version model has been uploaded to huggingface at https://huggingface.co/WANGNingroci/RNAErnie and the training framework/tokenization is located at https://github.com/CatIIIIIIII/RNAErnie2. (**NOTE**: the tokenization is a little different from the original paddle implementation). Moreover, [Multimolecule](https://github.com/DLS5-Omics/multimolecule) are implementing current most powerful RNA language model with transformers and pytorch. Our model also could be accessed at [https://huggingface.co/multimolecule/rnaernie](https://huggingface.co/multimolecule/rnaernie).

- 2024.05.13: 🎉🎉 Our paper has been published at [https://www.nature.com/articles/s42256-024-00836-4](https://www.nature.com/articles/s42256-024-00836-4).
  
- 2024.04.20: 🎉🎉 RNAErnie has been accepted by Nature Machine Intelligence! The paper will be released soon.

- 2024.03.21: Add DOI and citation.

- 2024.01.26: Add ad-hoc pre-training with additional classification task.

- 2024.01.23: Integrate AUC metric in base_classes.py for simpler usage; Add content and update log section in README.md.

If you have any questions, feel free to contact us by email: `wangning.roci@gmail.com`.

## Installation

<!-- ### Use Docker Image (Strongly Recommended) -->

### Create Environment with Conda

First, download the repository and create the environment.

```bash
git clone https://github.com/CatIIIIIIII/RNAErnie.git
cd ./RNAErnie
conda env create -f environment.yml
```

Then, activate the "RNAErnie" environment.

```bash
conda activate RNAErnie
```

or you could 

### Run in Docker
#### Step1: Prepare code
First clone the repository:
```bash
git clone https://github.com/CatIIIIIIII/RNAErnie.git
```

#### Step2: Prepare running environment
Here we provide two ways to load the docker image.

**[Option1]**
You can directly access the docker image using this link:
```bash
https://hub.docker.com/r/nwang227/rnaernie
```

After docker sign in, you could pull the docker image using the following command:
```bash
sudo docker pull nwang227/rnaernie:1.1
```

**NOTE**:
1. If you encounter the error`unauthorized: authentication required`, this means that you haven't logged in your docker account to access docker hub.
 - Sign up a docker account
 - Login with `sudo docker login -u username --password-stdin`
 - Then try to pull the image again.

2. If you encounter the error `Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docke daemon running?`, this means that you haven't started the docker service.
    - Start the docker service with `systemctl start docker`
    - Then try to run the container again.

**[Option2]**
Or you can download the image tar from [Google Drive](https://drive.google.com/file/d/1Lkgw7w9xGZQ02PnU3yk0cn1V9om2yfd3) or use the url as follow

```bash
https://drive.google.com/file/d/1Lkgw7w9xGZQ02PnU3yk0cn1V9om2yfd3
```

and load by

```bash
sudo docker load --input rnaernie-1.1.tar
```

#### Step3: Run 
Run the container with data volumn mounted:
```bash
sudo docker run --gpus all --name rnaernie_docker -it -v $PWD/RNAErnie:/home/ nwang227/rnaernie:1.1 /bin/bash
```

**TODO**:
For python version conflict, RNA secondary structure prediction task is not available in docker image. We will fix in the future.

## Pre-training

### 1. Data Preparation

You can download my selected (nts<512) pretraining dataset from [Google Drive](https://drive.google.com/file/d/17nGJz0NW-Kd_Z3wAFhzeW5AUNAka6Yed/view?usp=sharing) or from [RNAcentral](https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/rnacentral_active.fasta.gz) and place the `.fasta` files in the `./data/pre_random` folder.

Then, you can use the following command to generate the pre-training data:

### 2. Pre-training

Pretrain RNAErnie on selected RNAcentral datasets (nts<=512) with the following command:

```bash
python run_pretrain.py \
    --output_dir=./output \
    --per_device_train_batch_size=50 \
    --learning_rate=0.0001 \
    --save_steps=1000
```

To use multi-gpu training, you can add the following arguments:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m paddle.distributed.launch run_pretrain.py 
```

where `CUDA_VISIBLE_DEVICES` specifies the GPU ids you want to use.

### 3. Download Pre-trained Models

Our pre-trained model with BERT, ERNIE and MOTIF masking strategies could be downloaded from [Google Drive](https://drive.google.com/drive/folders/1Ls5k7hv83BLRTznB4XcegIa2yKkU40Ls?usp=drive_link) and place the `.pdparams` and `.json` files in the `./output/BERT,ERNIE,MOTIF,PROMPT` folder.

### 4. Visualization

You can visualize the pre-training process with the following command:

```bash
visualdl --logdir ./output/BERT,ERNIE,MOTIF,PROMPT/runs/you_date/ 
```

### 5. Extract RNA Sequence Embeddings

Then you could extract embeddings of given RNA sequences or from `.fasta` file with the following codes:

```python
import paddle
from rna_ernie import BatchConverter
from paddlenlp.transformers import ErnieModel

# ========== Set device
paddle.set_device("gpu")

# ========== Prepare Data
data = [
    ("RNA1", "GGGUGCGAUCAUACCAGCACUAAUGCCCUCCUGGGAAGUCCUCGUGUUGCACCCCU"),
    ("RNA2", "GGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCA"),
    ("RNA3", "CGAUUCNCGUUCCC--CCGCCUCCA"),
]
# data = "./data/ft/seq_cls/nRC/test.fa"

# ========== Batch Converter
batch_converter = BatchConverter(k_mer=1,
                                  vocab_path="./data/vocab/vocab_1MER.txt",
                                  batch_size=256,
                                  max_seq_len=512)

# ========== RNAErnie Model
rna_ernie = ErnieModel.from_pretrained("output/BERT,ERNIE,MOTIF,PROMPT/checkpoint_final/")
rna_ernie.eval()

# call batch_converter to convert sequences to batch inputs
for names, _, inputs_ids in batch_converter(data):
    with paddle.no_grad():
        # extract whole sequence embeddings
        embeddings = rna_ernie(inputs_ids)[0].detach()
        # extract [CLS] token embedding
        embeddings_cls = embeddings[:, 0, :]
```

## Downstream Tasks

### RNA sequence classification

#### 1. Data Preparation

You can download training data from [Google Drive](https://drive.google.com/drive/folders/1flh2rXiMKIreHE2l4sbjMmwAqfURj4vv?usp=sharing) and place them in the `./data/ft/seq_cls` folder. Three datasets (nRC, lncRNA_H, lncRNA_M) are available for this task.

#### 2. Fine-tuning

Fine-tune RNAErnie on RNA sequence classification task with the following command:

```bash
python run_seq_cls.py \
    --dataset=nRC \
    --dataset_dir=./data/ft/seq_cls \
    --model_name_or_path=./output/BERT,ERNIE,MOTIF,PROMPT/checkpoint_final \
    --train=True \
    --batch_size=50 \
    --num_train_epochs=100 \
    --learning_rate=0.0001 \
    --output=./output_ft/seq_cls
```

Moreover, to train on long ncRNA classification tasks, change augument `--dataset` to `lncRNA_M` or `lncRNA_H`, and you can add the `--use_chunk=True` argument to chunk and ensemble the whole sequence.

To use two-stage fine-tuning, you can add the `--two_stage=True` argument.

#### 3. Evaluation

Or you could download our weights of RNAErnie on sequence classification tasks from [Google Drive](https://drive.google.com/drive/folders/1v7Wx6cOd7_3EGtxTAMWMjtPTwocL5w6-?usp=sharing) and place them in the `./output_ft/seq_cls` folder.

Then you could evaluate the performance with the following codes:  

```bash
python run_seq_cls.py \
    --dataset=nRC \
    --dataset_dir=./data/ft/seq_cls \
    --model_name_or_path=./output/BERT,ERNIE,MOTIF,PROMPT/checkpoint_final \
    --model_path=./output_ft/seq_cls/nRC/BERT,ERNIE,MOTIF,PROMPT/model_state.pdparams \
    --train=False \
    --batch_size=50
```

To evaluate two-stage procedure, you can add the `--two_stage=True` argument and change the `--model_path` to `./output_ft/seq_cls/nRC/BERT,ERNIE,MOTIF,PROMPT,2`.

### RNA RNA interaction prediction

#### 1. Data Preparation

You can download training data from [Google Drive](https://drive.google.com/drive/folders/1iZK3-rw0QCyustOEUaEII8t2wXS2SwFc?usp=sharing) and place them in the `./data/ft/rr_inter` folder.

#### 2. Fine-tuning

Fine-tune RNAErnie on RNA-RNA interaction task with the following command:

```bash
python run_rr_inter.py \
    --dataset=MirTarRAW \
    --dataset_dir=./data/ft/rr_inter \
    --model_name_or_path=./output/BERT,ERNIE,MOTIF,PROMPT/checkpoint_final \
    --train=True \
    --batch_size=256 \
    --num_train_epochs=100 \
    --lr=0.001 \
    --output=./output_ft/rr_inter
```

#### 3. Evaluation

Or you could download our weights of RNAErnie on RNA RNA interaction tasks from [Google Drive](https://drive.google.com/drive/folders/1DgNAngxNlqmlVrHUqn4ygzgzXt0DA4Xy?usp=sharing) and place them in the `./output_ft/rr_inter` folder.

Then you could evaluate the performance with the following codes:  

```bash
python run_rr_inter.py \
    --dataset=MirTarRAW \
    --dataset_dir=./data/ft/rr_inter \
    --model_name_or_path=./output/BERT,ERNIE,MOTIF,PROMPT/checkpoint_final \
    --model_path=./output_ft/rr_inter/MirTarRAW/BERT,ERNIE,MOTIF,PROMPT \
    --train=False \
    --batch_size=256
```

### RNA secondary structure prediction

#### 1. Data Preparation

You can download training data from [Google Drive](https://drive.google.com/drive/folders/1XUBVXAUyIB6NqWmwEdLLlnWFaoU_l3XN?usp=sharing) and unzip and place them in the `./data/ft/ssp` folder. Two tasks (RNAStrAlign-ArchiveII, bpRNA1m) are available for this task.

#### 2. Adaptation

Adapt RNAErnie on RNA secondary structure prediction task with the following command:

```bash
python run_ssp.py \
    --task_name=RNAStrAlign \
    --dataset_dir=./data/ft/ssp \
    --model_name_or_path=./output/BERT,ERNIE,MOTIF,PROMPT/checkpoint_final \
    --train=True \
    --num_train_epochs=50 \
    --lr=0.001 \
    --output=./output_ft/ssp
```

**Note**: we use `interface.*.so` compiled from [mxfold2](https://github.com/mxfold/mxfold2). If you system could not run the `interface.*.so` file, you could download the source code from [here](https://github.com/mxfold/mxfold2/releases/download/v0.1.2/mxfold2-0.1.2.tar.gz) and compile it by yourself. Then copy the generated `interface.*.so` file to `./` path.

#### 3. Evaluation

Or you could download our weights of RNAErnie on RNA secondary structure prediction tasks from [Google Drive](https://drive.google.com/drive/folders/1UljS7YvDdYvWgmR7R5EdBG0OPPbkKoaB?usp=sharing) and place them in the `./output_ft/ssp` folder.

Then you could evaluate the performance with the following codes:  

```bash
python run_ssp.py \
    --task_name=RNAStrAlign \
    --dataset_dir=./data/ft/ssp \
    --train=False
```

## Baselines
We also implement other BERT-like large-scale pre-trained RNA language models for comparison, see here: https://github.com/CatIIIIIIII/RNAErnie_baselines.

## Citation
If you use the code or the data for your research, please cite our paper as follows:
```
@Article{Wang2024,
author={Wang, Ning
and Bian, Jiang
and Li, Yuchen
and Li, Xuhong
and Mumtaz, Shahid
and Kong, Linghe
and Xiong, Haoyi},
title={Multi-purpose RNA language modelling with motif-aware pretraining and type-guided fine-tuning},
journal={Nature Machine Intelligence},
year={2024},
month={May},
day={13},
issn={2522-5839},
doi={10.1038/s42256-024-00836-4},
url={https://doi.org/10.1038/s42256-024-00836-4}
}
```

## Future work
We pretrained  model from scractch with additional classification head appended to '[CLS]' token. The total loss function is
$L = L_{MLM}+\alpha L_{CLS}$
where $L_{MLM}$ is mask language model loss and $L_{CLS}$ is classification loss and we set the balance coefficient $\alpha$ as $0.1$. Other settings are kept same with our original RNAErnie pre-training procedure.

The pre-trained model could be downloaded from [Google Drive](https://drive.google.com/drive/folders/13Wmw_1hM-iPdvIhJUPxtH-sR92PUvCVr?usp=sharing) and place the `.pdparams` and `.json` files in the `./output/BERT,ERNIE,MOTIF,ADHOC` folder. Moreover, original pre-trained RNAErnie weight at 1 epoch could be obtained from [Google Drive](https://drive.google.com/drive/folders/1nHyHbnBpSgMVYvTU4T7LjTmmWDE2KiKY?usp=sharing).
