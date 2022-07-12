from typing import Tuple, List, Dict, cast, Optional
import torch
from neural_chat.actor import DictActor
from neural_chat.critic import DoubleQCritic
from neural_chat.craigslist import CraigslistData
from neural_chat.gpt2 import GPT2, format_dialog
from .base import Model
from neural_chat.critic import DoubleQCritic
import torch.distributions as pyd
import termcolor
import itertools
import copy
import neural_chat.craigslist as cg
import torch


def load_model(
    checkpoint_file: str, has_actor: bool
) -> Tuple[DoubleQCritic, Optional[DictActor]]:
    obs_spec, act_spec = CraigslistData.obs_spec, CraigslistData.act_spec
    with open(checkpoint_file, "rb") as f:
        snapshot = torch.load(f)
        critic = DoubleQCritic(obs_spec, act_spec, [256, 256])
        critic.load_state_dict(**snapshot["algo"]["critic"])
        critic.cuda()
        critic.eval()

        actor = None
        if has_actor:
            actor = DictActor(obs_spec, act_spec, [256, 256])
            actor.load_state_dict(**snapshot["algo"]["actor"])
            actor.cuda()
            actor.eval()
    return critic, actor


class Ours(Model):
    def __init__(
        self,
        gpt_dir: str,
        checkpoint_file: str,
        num_outputs: int,
        temperature: float = 10,
        has_actor: bool = False,
    ):
        self.gpt = GPT2(gpt_dir)
        c, a = load_model(checkpoint_file, has_actor)
        self.critic = cast(DoubleQCritic, c)
        if a is not None:
            self.actor = cast(DictActor, a)
        self.temperature = temperature
        self.num_outputs = num_outputs

        # state tracking
        self.dialog: List[cg.EventType] = []

    def _embeddings(self, events: List[List[cg.EventType]]) -> torch.Tensor:
        agents = itertools.cycle([cg.Agent.BUYER, cg.Agent.SELLER])
        formatted = [
            format_dialog(self.scenario, list(zip(agents, map(str, event))))
            for event in events
        ]
        return self.gpt.embedding(formatted)

    def _buyer_seller_price(self, events: List[cg.EventType]) -> Tuple[float, float]:
        pt = cg.price_parser.PriceParser(
            self.scenario.list_price,
            0.8 * self.scenario.list_price,
            self.scenario.list_price,
        )
        agent = cg.Agent.BUYER

        for ev in events:
            pt.update_event(ev, agent)
            agent = agent.other_agent()

        agent = agent.other_agent()
        bp = pt.get_price(agent.BUYER)
        sp = pt.get_price(agent.SELLER)
        assert bp is not None and sp is not None, "impossible"

        bp, sp = map(lambda x: x / self.scenario.list_price, (bp, sp))
        return bp, sp

    def _obs(self, events: List[cg.EventType], repeat: int = 1) -> dict:
        # get buyer and seller price
        bp, sp = self._buyer_seller_price(events)

        # calculate embedding
        embedding = self._embeddings([events]).to("cuda")

        # make final feed dict
        tensor = lambda x: torch.tensor([[x]]).float().to("cuda")
        obs = {
            "utterance": embedding,
            "buyer_price": tensor(bp),
            "seller_price": tensor(sp),
            "action_type": tensor(cg.event_to_int(events[-1])),
        }
        obs = {k: v.repeat(repeat, 1) for k, v in obs.items()}
        return obs

    @staticmethod
    def _str_to_event(candidate: str) -> cg.EventType:
        """For offer types, this DOES NOT fill in the price."""
        if candidate == "deal":
            return cg.Accept()
        elif candidate == "no deal":
            return cg.Reject()
        elif candidate == "offer":
            return cg.Offer(0)
        elif candidate == "quit":
            return cg.Quit()
        return cg.Message(candidate)

    def _candidates(
        self, events: List[cg.EventType]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], List[cg.EventType]]:
        # calculate obs
        obs = self._obs(events, repeat=self.num_outputs * 5)

        # calculate candidates
        candidates = self.gpt.generate(
            self.scenario, list(map(str, events)), num_outputs=self.num_outputs
        )
        candidates = [self._str_to_event(c) for c in candidates]
        embeddings = self._embeddings(
            [events + [c] for c in candidates]
        ).repeat_interleave(5, dim=0)
        action_types = (
            torch.tensor([cg.event_to_int(c) for c in candidates])
            .unsqueeze(1)
            .float()
            .repeat_interleave(5, dim=0)
            .cuda()
        )

        # calculate prices
        sp = obs["seller_price"].reshape(-1, 5)[:, 0, None]
        bp = obs["buyer_price"].reshape(-1, 5)[:, 0, None]
        avg = (sp + bp) / 2.0
        price_candidates = torch.cat(
            [
                bp,
                (avg + bp) / 2,
                avg,
                (avg + sp) / 2,
                sp,
            ],
            dim=1,
        ).reshape(-1, 1)

        # make actions
        act = {
            "utterance": embeddings.repeat_interleave(5, dim=0),
            "price": torch.clamp(price_candidates, 0, 3),
            "action_type": action_types.repeat_interleave(5, dim=0),
        }

        # return all candidates
        candidates = list(sum([(c,) * 5 for c in candidates], ()))
        return obs, act, candidates

    def actions(self, events: List[cg.EventType]) -> List[cg.EventType]:

        # calculate the prices
        obs, act, candidates = self._candidates(events)

        # fill in the prices
        new_candidates = []
        for candidate, price in zip(candidates, act["price"].squeeze().tolist()):
            if isinstance(candidate, cg.Offer):
                new_candidates.append(cg.Offer(price * self.scenario.list_price))
            elif isinstance(candidate, cg.Message):
                bp = float(obs["buyer_price"][0].item())
                new_candidates.append(
                    cg.Message(
                        candidate.utterance.replace(
                            "$PRICE", str(price * self.scenario.list_price)
                        ).replace("$PARTNER_PRICE", str(bp * self.scenario.list_price))
                    )
                )
            else:
                new_candidates.append(copy.copy(candidate))

        return new_candidates

    def _q_vals(self, events: List[cg.EventType], candidates: List[cg.EventType]):
        obs = self._obs(events, len(candidates))
        act = {
            "utterance": self._embeddings([events + [c] for c in candidates]),
            "price": torch.tensor(
                [self._buyer_seller_price(events + [c])[1] for c in candidates]
            )
            .unsqueeze(1)
            .cuda(),
            "action_type": torch.tensor([cg.event_to_int(ev) for ev in candidates])
            .unsqueeze(1)
            .cuda(),
        }
        return self.critic(obs, act).squeeze().cpu()

    def reset(self, scenario: cg.Scenario) -> None:
        self.scenario = scenario
        self.dialog = []

    def response(self, ev: cg.EventType, debug: bool = False) -> cg.EventType:
        with torch.no_grad():
            # special handling of actor case
            if hasattr(self, "actor"):
                candidates = self.gpt.generate(
                    self.scenario,
                    list(map(str, self.dialog)),
                    num_outputs=self.num_outputs,
                )
                candidate = candidates[0]
                obs = self._obs(self.dialog, repeat=1)
                actor_action = self.actor.action_with_log_prob(obs)
                price = actor_action[0]["price"].squeeze().detach().cpu().numpy()
                bp = float(obs["buyer_price"][0].item())
                ev = self._str_to_event(
                    candidate.replace(
                        "$PRICE", str(price * self.scenario.list_price)
                    ).replace("$PARTNER_PRICE", str(bp * self.scenario.list_price))
                )
                if isinstance(ev, cg.Offer):
                    ev = cg.Offer(price * self.scenario.list_price)
                return ev

            self.dialog.append(ev)

            # get candidate actions
            candidates = self.actions(self.dialog)

            # get ranking
            qs = self._q_vals(self.dialog, candidates)
            ranking = torch.argsort(qs)

            # calculate index by sampling
            index = (
                pyd.Categorical(logits=qs.squeeze() * self.temperature).sample().item()
            )

            # print out ranking
            if debug:
                for r in ranking:
                    out = "Q:{:7.6f}\t|{}".format(qs[r].item(), candidates[r])
                    if index == r:
                        out = termcolor.colored(out, "red")
                    print(out)

            # return response
            ev = candidates[cast(int, index)]
            self.dialog.append(ev)
            return ev
