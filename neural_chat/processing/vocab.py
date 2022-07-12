import os
from neural_chat import utils


def load_vocab(vocab_file=None):
    if vocab_file is None:
        vocab_file = os.path.join(utils.COCOA_DATA_DIR, "vocab.txt")
    with open(vocab_file, "r") as f:
        words = [word.strip() for word in f]
    return Vocab(words)


def rmap(l, fn):
    all_mapped = []
    for elem in l:
        if isinstance(elem, list):
            mapped_elem = rmap(elem, fn)
        else:
            mapped_elem = fn(elem)
        all_mapped.append(mapped_elem)
    return all_mapped


class Vocab(object):
    def __init__(
        self, word_list, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>"
    ):
        self._word_list = word_list
        self._word_2_idx = {
            self._word_list[idx]: idx for idx in range(len(self._word_list))
        }

        # Special characters
        self.unk = len(self._word_list)
        self._word_2_idx[unk_token] = self.unk
        self._word_list.append(unk_token)

        self.pad = len(self._word_list)
        self._word_2_idx[pad_token] = self.pad
        self._word_list.append(pad_token)

        self.eos = len(self._word_list)
        self._word_2_idx[eos_token] = self.eos
        self._word_list.append(eos_token)

    def __len__(self):
        return len(self._word_list)

    def indices2words(self, indices):
        return rmap(indices, lambda idx: self._word_list[idx])

    def words2indices(self, words):
        return rmap(words, lambda word: self._word_2_idx.get(word, self.unk))
