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

# call batch_converter to convert sequences to batch inputs
for names, _, inputs_ids in batch_converter(data):
    with paddle.no_grad():
        # extract whole sequence embeddings
        embeddings = rna_ernie(inputs_ids)[0].detach()
        # extract [CLS] token embedding
        embeddings_cls = embeddings[:, 0, :]
