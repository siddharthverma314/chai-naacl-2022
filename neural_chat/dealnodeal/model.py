from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from torch import nn
import torch
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers.file_utils import ModelOutput
from transformers import generation_utils, PreTrainedModel


class DNDConfig(PretrainedConfig):
    model_dim: int
    token_dim: int
    buyer_tok_id: int
    seller_tok_id: int


@dataclass
class DNDMem(dict):
    utterance_embed: torch.Tensor
    context_embed: torch.Tensor

    def reorder(self, idx: torch.Tensor) -> DNDMem:
        return DNDMem(
            utterance_embed=self.utterance_embed.index_select(0, idx),
            context_embed=self.context_embed.index_select(1, idx),
        )


# for TorchScript to work, this has to be a separate function
@torch.jit.script
def buyer_seller_mask(
    input_ids: torch.Tensor, buyer_tok_id: int, seller_tok_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    state = torch.zeros_like(input_ids)
    for pos in range(1, input_ids.shape[1]):
        cur_pos = input_ids[:, pos]
        prev_pos = input_ids[:, pos - 1]
        is_exit = (cur_pos == buyer_tok_id) | (cur_pos == seller_tok_id)
        is_buyer = (prev_pos == buyer_tok_id) & ~is_exit
        is_seller = (prev_pos == seller_tok_id) & ~is_exit

        is_write = is_exit | is_buyer | is_seller
        buf = torch.zeros_like(state[:, 0])
        new_state = is_exit * buf + is_buyer * (buf + 1) + is_seller * (buf + 2)

        state[:, pos] = state[:, pos - 1] * ~is_write + new_state * is_write

    return state == 1, state == 2


class DND(PreTrainedModel, generation_utils.GenerationMixin):
    config_class = DNDConfig
    base_model_prefix = "dnd"

    def __init__(self, config: DNDConfig):
        """GRU Model as defined in Deal No Deal."""
        PreTrainedModel.__init__(self, config)
        generation_utils.GenerationMixin.__init__(self)

        self.embed_context = nn.Embedding(config.token_dim, config.model_dim)
        self.embed_utterance = nn.Embedding(config.token_dim, config.model_dim)
        self.gru_context = nn.GRU(config.model_dim, config.model_dim, batch_first=True)
        self.gru_utterance = nn.GRU(
            config.model_dim, config.model_dim, batch_first=True
        )
        self.final = nn.Linear(config.model_dim, config.token_dim)

        self.buyer_tok_id = config.buyer_tok_id
        self.seller_tok_id = config.seller_tok_id

    @staticmethod
    def _reorder_cache(past: DNDMem, beam_idx: torch.Tensor) -> DNDMem:
        return past.reorder(beam_idx)

    @staticmethod
    def prepare_inputs_for_generation(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_input_ids: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        batch_size = input_ids.shape[0]
        context_input_ids = context_input_ids.repeat(batch_size, 1)
        return {
            "utterance_input_ids": input_ids,
            "utterance_attention_mask": attention_mask,
            "context_input_ids": context_input_ids,
            **kwargs,
        }

    def forward(
        self,
        context_input_ids: torch.Tensor,
        utterance_input_ids: torch.Tensor,
        utterance_attention_mask: torch.Tensor,
        use_cache: bool = False,
        past: Optional[DNDMem] = None,
        **_,
    ) -> ModelOutput:
        """Calculates embeddings for next word under model."""
        if use_cache and past is not None:
            # we have cached model outputs
            context_embed: torch.Tensor = past.context_embed
            utterance_embed: torch.Tensor = past.utterance_embed
            utterance_input_ids = utterance_input_ids[:, -1:]
            utterance_attention_mask = utterance_attention_mask[:, -1:]
            hx = utterance_embed[:, -1:, :].transpose(0, 1)
        else:
            # hidden state is from context
            _, context_embed = self.gru_context.forward(
                self.embed_context(context_input_ids)
            )
            hx = context_embed

        # next, calculate utterance
        utterance_embed, _ = self.gru_utterance.forward(
            self.embed_utterance(utterance_input_ids), hx=hx.contiguous()
        )

        # calculate logits
        utterance_logits = self.final(utterance_embed)

        # calculate loss
        logits = utterance_logits[:, :-1, :]
        labels = utterance_input_ids[:, 1:]
        mask = utterance_attention_mask[:, 1:]
        nll = F.cross_entropy(
            logits.flatten(end_dim=1), labels.flatten(), reduction="none"
        ).reshape(labels.shape)
        masked_mean = lambda mask: nll.sum(axis=1) / mask.sum(axis=1)

        # get buyer and seller utterance masks
        buyer_mask, seller_mask = buyer_seller_mask(
            utterance_input_ids, self.buyer_tok_id, self.seller_tok_id
        )
        buyer_mask = buyer_mask[:, 1:]
        seller_mask = seller_mask[:, 1:]

        return ModelOutput(
            loss=masked_mean(mask).mean(),
            logits=utterance_logits,
            nll_buyer=masked_mean(mask * buyer_mask),
            nll_seller=masked_mean(mask * seller_mask),
            mems=DNDMem(
                context_embed=context_embed,
                utterance_embed=utterance_embed,
            ),
        )
