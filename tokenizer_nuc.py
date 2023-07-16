"""
This module builds RNA nucleotide tokenizer.

Author: wangning(wangning.roci@gmail.com)
Date  : 2022/8/8 2:43 PM
"""

# built-in modules
import os
import io
# paddle modules
from paddlenlp.transformers import BasicTokenizer, PretrainedTokenizer
from paddlenlp.data.vocab import Vocab


class NUCTokenizer(PretrainedTokenizer):
    """
    Constructs an ERNIE tokenizer. It uses a basic tokenizer to do punctuation
    splitting, lower casing and so on, and follows a WordPiece tokenizer to
    tokenize as subwords.

    This tokenizer inherits from :class:`~paddlenlp.transformers.tokenizer_utils.PretrainedTokenizer`
    which contains most of the main methods. For more information regarding those methods,
    please refer to this superclass.

    Args:
        vocab_file (str):
            The vocabulary file path (ends with '.txt') required to instantiate
            a `WordpieceTokenizer`.
        do_lower_case (str, optional):
            Whether or not to lowercase the input when tokenizing.
            Defaults to`True`.
        unk_token (str, optional):
            A special token representing the *unknown (out-of-vocabulary)* token.
            An unknown token is set to be `unk_token` inorder to be converted to an ID.
            Defaults to "[UNK]".
        sep_token (str, optional):
            A special token separating two different sentences in the same input.
            Defaults to "[SEP]".
        pad_token (str, optional):
            A special token used to make arrays of tokens the same size for batching purposes.
            Defaults to "[PAD]".
        cls_token (str, optional):
            A special token used for sequence classification. It is the last token
            of the sequence when built with special tokens. Defaults to "[CLS]".
        mask_token (str, optional):
            A special token representing a masked token. This is the token used
            in the masked language modeling task which the model tries to predict the original unmasked ones.
            Defaults to "[MASK]".
        del_token (str, optional):
            A special token representing a deleted token. Defaults to "[DEL]".
        ind_token (str, optional):
            A special token representing a indication token. Defaults to "[IND]".

    Examples:
        .. code-block::

            from paddlenlp.transformers import ErnieTokenizer
            tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')

            encoded_inputs = tokenizer('He was a puppeteer')
            # encoded_inputs:
            # { 'input_ids': [1, 4444, 4385, 1545, 6712, 10062, 9568, 9756, 9500, 2],
            #  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
            # }

    """

    # resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    # pretrained_resource_files_map = None
    # pretrained_init_configuration = None

    def __init__(self, k_mer, vocab_file, do_lower_case=False, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]", del_token="[DEL]", ind_token="[IND]", label_tokens=None, normal_tokens_start=35):
        """init tokenizer

        Args:
            k_mer (int): k_mer for seperating RNA sequence
            vocab_file (str): vocabulary file path
            do_lower_case (bool, optional): whether convert to lower cases. Defaults to False.
            unk_token (str, optional): unknown tokens.
            pad_token (str, optional): padding tokens.
            cls_token (str, optional): begin og sentence tokens. Defaults to "[CLS]".
            sep_token (str, optional): end of sentence tokens.
            mask_token (str, optional): mask tokens. Defaults to "[MASK]".
            del_token (str, optional): deletion tokens. Defaults to "[DEL]".
            label_tokens (str, optional): label tokens.
            ind_token (str, optional): indication tokens. Defaults to "[IND]".
            normal_tokens_start (int, optional): sentences tokens starting index. Defaults to 35.

        Raises:
            ValueError: _description_
        """

        if not os.path.isfile(vocab_file):
            raise ValueError("Can't find a vocabulary file at path '{}'. "
                             "To load the vocabulary from a pretrained model please use "
                             "`tokenizer = ErnieTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        self.k_mer = k_mer
        self.vocab_file = vocab_file
        # self.do_lower_case = do_lower_case

        extra_label_tokens = []
        if label_tokens:
            for k, v in label_tokens.items():
                if v == "-":
                    extra_label_tokens.append(k.replace("_", ""))

        self.vocab = self.load_vocabulary(vocab_file,
                                          unk_token=unk_token,
                                          pad_token=pad_token,
                                          eos_token=sep_token,
                                          label_tokens=extra_label_tokens)

        self.vocab_id_list = list(self.vocab.idx_to_token.keys())
        self.normal_vocab_id_list = self.vocab_id_list[normal_tokens_start:]
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        # self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab,
        #                                               unk_token=unk_token)
        self.unk_token = unk_token
        self.del_token = del_token
        self.ind_token = ind_token
        self.sep_token = sep_token

    @property
    def vocab_size(self):
        """
        Return the size of vocabulary.

        Returns:
            int: The size of vocabulary.
        """
        return len(self.vocab)

    def load_vocabulary(self, filepath, unk_token=None, pad_token=None, bos_token=None, eos_token=None, label_tokens=None, **kwargs):
        """load vocabulary from file

        Args:
            filepath (str, optional): vocabulary file path
            unk_token (str, optional): unknown tokens.
            pad_token (str, optional): padding tokens.
            bos_token (str, optional): begin of sentence tokens.
            eos_token (str, optional): end of sentence tokens.
            label_tokens (str, optional): label tokens.

        Returns:
            Vocab: vocabulary instance
        """
        token_to_idx = {}
        with io.open(filepath, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                token = line.rstrip('\n')
                token_to_idx[token] = int(index)

        for extra_label in label_tokens:
            index += 1
            token_to_idx[extra_label] = index

        vocab = Vocab.from_dict(token_to_idx,
                                unk_token=unk_token,
                                pad_token=pad_token,
                                bos_token=bos_token,
                                eos_token=eos_token,
                                **kwargs)
        return vocab

    def _tokenize(self, text):
        """
        End-to-end tokenization for ERNIE models.

        Args:
            text (str): The text to be tokenized.

        Returns:
            List[str]: A list of string representing converted tokens.
        """
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            split_tokens.append(token)

        return split_tokens

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        r"""
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens.

        An Ernie sequence has the following format:

        - single sequence:      ``[CLS] X [SEP]``
        - pair of sequences:        ``[CLS] A [SEP] B [SEP]``

        Args:
            token_ids_0 (List[int]):
                List of IDs to which the special tokens will be added.
            token_ids_1 (List[int], optional):
                Optional second list of IDs for sequence pairs.
                Defaults to `None`.

        Returns:
            List[int]: List of input_id with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        _cls = [self.cls_token_id]
        _sep = [self.sep_token_id]
        return _cls + token_ids_0 + _sep + token_ids_1 + _sep
