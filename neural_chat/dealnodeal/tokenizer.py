from __future__ import annotations
from typing import Tuple, Optional, Union
import transformers
import itertools
from neural_chat.dealnodeal.utils import iter_utterance, iter_context
import tempfile
import os
import json
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace, Digits, Sequence
from tokenizers.normalizers import Lowercase
from tokenizers.trainers import WordPieceTrainer
from tokenizers import decoders


class DNDTokenizer(transformers.PreTrainedTokenizerFast):
    def __init__(
        self, tokenizer_file=None, eos_token="<eos>", pad_token="<pad>", **kwargs
    ):
        super().__init__(
            tokenizer_file=tokenizer_file,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        files = self.backend_tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    @classmethod
    def _from_pretrained(cls, path: dict, *args, **kwargs):
        with open(path["tokenizer_config_file"]) as f:
            config = json.load(f)
            return cls(path["tokenizer_file"], **config)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        legacy_format: bool = False,
        filename_prefix: Optional[str] = None,
    ) -> Tuple[str]:
        return transformers.PreTrainedTokenizerFast.save_pretrained(
            self,
            save_directory=save_directory,
            legacy_format=legacy_format,
            filename_prefix=filename_prefix,
        )

    @classmethod
    def train_tokenizer(cls) -> DNDTokenizer:
        token = Tokenizer(WordPiece(unk_token="[UNK]"))
        token.pre_tokenizer = Sequence([Whitespace(), Digits()])
        token.normalizer = Lowercase()
        token.add_special_tokens(["<eof>", "[UNK]"])
        token.enable_padding(pad_token="<pad>")
        token.add_special_tokens(["<buyer>", "<seller>"])
        token.decoder = decoders.WordPiece()

        dataset = itertools.chain(iter_utterance(), iter_context())
        filename = tempfile.mktemp()
        with open(filename, "w") as f:
            for line in dataset:
                f.write(line)
        token.train(
            WordPieceTrainer(
                special_tokens=["<eof>", "<pad>", "<buyer>", "<seller>", "[UNK]"],
                end_of_word_suffix="_",
            ),
            [filename],
        )

        token.save(filename)
        dnd_token = cls(filename)
        os.remove(filename)

        return dnd_token
