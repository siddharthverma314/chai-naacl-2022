from typing import List, Optional, Union, Tuple

import torch
from .processing import format_dialog
import neural_chat.craigslist as cg
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)


class GPT2(torch.nn.Module):
    def __init__(
        self,
        gpt2_type_or_checkpoint: str = "gpt2-medium",
        device: Union[torch.device, str] = "cuda",
    ):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type_or_checkpoint)
        self.tokenizer.pad_token = " "  # newline token
        self.model = GPT2LMHeadModel.from_pretrained(gpt2_type_or_checkpoint)

        self.model.eval()
        self.model.to(device)
        self.device = device

    def _encode(self, sent, **kwargs):
        output = self.tokenizer(
            sent,
            return_tensors="pt",
            **kwargs,
        )
        return {k: v.to(self.device) for k, v in output.items()}

    def _encode_dialog(self, context, dialog):
        return self._encode(format_dialog(context, dialog) + "<sep>Seller:")[
            "input_ids"
        ]

    def _decode(self, sent):
        return self.tokenizer.decode(sent.squeeze().tolist())

    def generate_fast(
        self,
        context: cg.Scenario,
        dialog: List[Tuple[cg.Agent, str]],
        num_last: int = 2,
        num_extra_beams: int = 2,
        **kwargs,
    ):
        dialog = dialog[-num_last:]
        encoded = self._encode_dialog(context, dialog)
        return self.model.generate(
            encoded,
            max_length=512,
            num_beams=kwargs["num_return_sequences"] + num_extra_beams,
            no_repeat_ngram_size=2,
            cache=True,
            **kwargs,
        )

    def generate_best_cut(
        self,
        context: cg.Scenario,
        dialog: List[Tuple[cg.Agent, str]],
        **kwargs,
    ):
        return self.generate_best(context, dialog[-2:], **kwargs)

    def generate_best(
        self,
        context: cg.Scenario,
        dialog: List[Tuple[cg.Agent, str]],
        **kwargs,
    ):
        encoded = self._encode_dialog(context, dialog)
        return self.model.generate(
            encoded,
            top_k=50,
            top_p=0.85,
            do_sample=True,
            max_length=1024,
            cache=True,
            num_beams=5,
            **kwargs,
        )

    def generate(
        self,
        context: cg.Scenario,
        dialog: List[str],
        agents: Optional[List[cg.Agent]] = None,
        num_outputs: int = 10,
        generation_fn="generate_best",
        **kwargs,
    ) -> List[str]:
        if agents is None:
            agents = [cg.Agent(i % 2) for i in range(len(dialog))]
        new_dialog = list(zip(agents, dialog))
        generation_fn = getattr(self, generation_fn)
        generated = generation_fn(
            context,
            new_dialog,
            agents=agents,
            early_stopping=True,
            eos_token_id=self.tokenizer.encode("<sep>")[0],
            pad_token_id=220,  # newline token
            num_return_sequences=num_outputs,
            **kwargs,
        )

        processed = []
        for g in generated:
            s = self._decode(g).strip().split("<sep>")[-2].strip()  # get last dialog
            if s.startswith("Seller:"):
                s = s[len("Seller:") :]  # get rid of Seller:
            s = s.strip()
            s = s.split("<|endoftext|>")[0].strip()  # stop next dialog from generating
            processed.append(s)
        return processed

    def embedding(self, sentences: List[str]) -> torch.Tensor:
        new_sentences = []
        for s in sentences:
            s = s.strip()
            if not s.endswith("<sep>"):
                s += "<sep>"
            new_sentences.append(s)

        encoded_input = self._encode(
            new_sentences,
            padding=True,
            truncation=True,
            max_length=128,
        )

        outputs = self.model.transformer(**encoded_input)
        token_embeddings = outputs[0]

        input_mask_expanded = (
            encoded_input["attention_mask"]
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
