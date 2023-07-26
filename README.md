# RNAErnie
Official implement of paper "Multi-purpose RNA Language Modeling with Motif-aware Pre-training and Type-guided Fine-tuning" with [paddlepaddle](https://github.com/PaddlePaddle/Paddle/tree/develop).

This repository contains codes and pre-trained models for RNAErnie, which leverages RNA motifs as **biological priors** and proposes a **motif-level random masking** strategy to enhance pre-training tasks. Furthermore, RNAErnie improves sequence classfication, RNA-RNA interaction prediction, and RNA secondary structure prediction by fine-tuning or adapating on downstream tasks with two-stage **type-guided** learning. Our paper will be published soon.

![Overview](./images/overview.png)


## Installation <a name="Installation"></a>
### Use Docker Image (Strongly Recommended) <a name="Use_docker_image"></a>

### Create Environment with Conda <a name="Setup_Environment"></a>

First, download the repository and create the environment.
```
git clone https://github.com/CatIIIIIIII/RNAErnie.git
cd ./RNAErnie
conda env create -f environment.yml
```
Then, activate the "RNA-FM" environment.
```
conda activate RNA-FM
```

## Pre-training <a name="Pre-training"></a>
### 1. Data Preparation <a name="Data_Preparation"></a>
You can download my pretraining dataset from [Google Drive](https://drive.google.com/file/d/17nGJz0NW-Kd_Z3wAFhzeW5AUNAka6Yed/view?usp=sharing) and place the `.fasta` files in the `./data/pre_random` folder.

### 2. Pre-training <a name="Pre-training"></a>
Pretrain RNAErnie on selected RNAcentral datasets (nts<=512) with the following command:
```
python run_pretrain.py \
    --dataset_dir=./data/pre_random \
    --k_mer=1 \
    --vocab_path=./data/vocab/ \
    --max_seq_length=512 \
    --pre_strategy=BERT,ERNIE,MOTIF \
    --num_groups=86 \
    --motif_files=ATtRACT,SpliceAid,Statistics \
    --output_dir=./output \
    --per_device_train_batch_size=50 \
    --learning_rate=0.0001 \
    --save_steps=1000
```
To use multi-gpu training, you can add the following arguments:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m paddle.distributed.launch run_pre
train.py 
```
where `CUDA_VISIBLE_DEVICES` specifies the GPU ids you want to use.
### 3. Download Pre-trained Models <a name="Download_Pre-trained_Models"></a>
Our pre-trained model with BERT, ERNIE and MOTIF masking strategies could be downloaded from [this gdrive link](https://drive.google.com/drive/folders/1Ls5k7hv83BLRTznB4XcegIa2yKkU40Ls?usp=drive_link) and place the `.pdparams` and `.json` files in the `./output/BERT,ERNIE,MOTIF,PROMPT` folder.

### 4. Extract RNA Sequence Embeddings <a name="RNA_sequence_embedding"></a>

Then you could extract embeddings of given RNA sequences or from `.fasta` file with the following codes:
```python
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

# call batch_converter to convert sequences to batch inputs
for names, _, inputs_ids in batch_converter(data):
    with paddle.no_grad():
        # extract whole sequence embeddings
        embeddings = rna_ernie(inputs_ids)[0].detach()
        # extract [CLS] token embedding
        embeddings_cls = embeddings[:, 0, :]
```


