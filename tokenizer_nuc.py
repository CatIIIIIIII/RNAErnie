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

    # def convert_tokens_to_string(self, tokens):
    #     r"""
    #     Converts a sequence of tokens (list of string) in a single string. Since
    #     the usage of WordPiece introducing `##` to concat subwords, also remove
    #     `##` when converting.

    #     Args:
    #         tokens (List[str]): A list of string representing tokens to be converted.

    #     Returns:
    #         str: Converted string from tokens.

    #     Examples:
    #         .. code-block::

    #             from paddlenlp.transformers import ErnieTokenizer
    #             tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')

    #             tokens = tokenizer.tokenize('He was a puppeteer')
    #             strings = tokenizer.convert_tokens_to_string(tokens)
    #             #he was a puppeteer

    #     """
    #     out_string = " ".join(tokens).replace(" ##", "").strip()
    #     return out_string

    # def num_special_tokens_to_add(self, pair=False):
    #     r"""
    #     Returns the number of added tokens when encoding a sequence with special tokens.

    #     Note:
    #         This encodes inputs and checks the number of added tokens, and is therefore not efficient.
    #         Do not put this inside your training loop.

    #     Args:
    #         pair (bool, optional):
    #             Whether the input is a sequence pair or a single sequence.
    #             Defaults to `False` and the input is a single sequence.

    #     Returns:
    #         int: Number of tokens added to sequences
    #     """
    #     token_ids_0 = []
    #     token_ids_1 = []
    #     return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))

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

    # def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
    #     r"""
    #     Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
    #     special tokens using the tokenizer ``encode`` methods.

    #     Args:
    #         token_ids_0 (List[int]):
    #             List of ids of the first sequence.
    #         token_ids_1 (List[int], optinal):
    #             Optional second list of IDs for sequence pairs.
    #             Defaults to `None`.
    #         already_has_special_tokens (str, optional):
    #             Whether or not the token list is already formatted with special tokens for the model.
    #             Defaults to `False`.

    #     Returns:
    #         List[int]:
    #             The list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
    #     """

    #     if already_has_special_tokens:
    #         if token_ids_1 is not None:
    #             raise ValueError("You should not supply a second sequence if the provided sequence of "
    #                              "ids is already formatted with special tokens for the model.")
    #         return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

    #     if token_ids_1 is not None:
    #         return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
    #     return [1] + ([0] * len(token_ids_0)) + [1]

    # def build_offset_mapping_with_special_tokens(self, offset_mapping_0, offset_mapping_1=None):
    #     r"""
    #     Build offset map from a pair of offset map by concatenating and adding offsets of special tokens.

    #     An ERNIE offset_mapping has the following format:

    #     - single sequence:      ``(0,0) X (0,0)``
    #     - pair of sequences:        ``(0,0) A (0,0) B (0,0)``

    #     Args:
    #         offset_mapping_ids_0 (List[tuple]):
    #             List of char offsets to which the special tokens will be added.
    #         offset_mapping_ids_1 (List[tuple], optional):
    #             Optional second list of wordpiece offsets for offset mapping pairs.
    #             Defaults to `None`.

    #     Returns:
    #         List[tuple]: A list of wordpiece offsets with the appropriate offsets of special tokens.
    #     """
    #     if offset_mapping_1 is None:
    #         return [(0, 0)] + offset_mapping_0 + [(0, 0)]

    #     return [(0, 0)] + offset_mapping_0 + [(0, 0)] + offset_mapping_1 + [(0, 0)]

    # def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
    #     r"""
    #     Create a mask from the two sequences passed to be used in a sequence-pair classification task.

    #     A ERNIE sequence pair mask has the following format:
    #     ::

    #         0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
    #         | first sequence    | second sequence |

    #     If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

    #     Args:
    #         token_ids_0 (List[int]):
    #             A list of `inputs_ids` for the first sequence.
    #         token_ids_1 (List[int], optional):
    #             Optional second list of IDs for sequence pairs.
    #             Defaults to `None`.

    #     Returns:
    #         List[int]: List of token_type_id according to the given sequence(s).
    #     """
    #     _sep = [self.sep_token_id]
    #     _cls = [self.cls_token_id]
    #     if token_ids_1 is None:
    #         return len(_cls + token_ids_0 + _sep) * [0]
    #     return len(_cls + token_ids_0 + _sep) * [0] + len(token_ids_1 + _sep) * [1]
