from typing import Dict, Any, cast, Optional, Tuple
from flatten_dict import flatten, unflatten
from .parse import (
    Craigslist,
    Event,
    Agent,
    Quit,
    Reject,
    Offer,
    Message,
    event_to_int,
    Accept,
)
import torch
import gym.spaces as sp
from enum import Enum
from torch.utils import data


class DoneType(Enum):
    NOT_DONE = 0
    SELLER_ACCEPT = 1
    SELLER_REJECT = 2
    BUYER_ACCEPT = 3
    BUYER_REJECT = 4


# spaces
price_box = sp.Box(0, 3, (1,))
embedding = sp.Box(-1.0, 1.0, (1024,))  # embedding size


REWARD_TYPE_UTILITY = "utility"
REWARD_TYPE_FAIRNESS = "fairness"
REWARD_TYPE_AGREEMENT = "agreement"


class CraigslistData(data.Dataset):

    obs_spec = sp.Dict(
        {
            "utterance": embedding,
            "seller_price": price_box,
            "buyer_price": price_box,
            "action_type": sp.Discrete(5),
        }
    )
    act_spec = sp.Dict(
        {
            "utterance": embedding,
            "price": price_box,
            "action_type": sp.Discrete(5),
        }
    )

    def __init__(
        self,
        path: str,
        embeddings_path: str,
        sentences_path: str,
        price_decrease_penalty=False,
        reward_type: str = REWARD_TYPE_UTILITY,
    ):
        self.cg = Craigslist(
            path, {"embeddings": embeddings_path, "sentences": sentences_path}
        )
        if reward_type == REWARD_TYPE_UTILITY:
            self.reward_handler = UtilityReward(price_decrease_penalty)
        elif reward_type == REWARD_TYPE_FAIRNESS:
            self.reward_handler = FairnessReward()
        elif reward_type == REWARD_TYPE_AGREEMENT:
            self.reward_handler = AgreementReward()
        else:
            raise ValueError("Bad reward type:", reward_type)

        # data
        self.data = []
        for ev in self.cg.events.values():
            data = self._format_data(ev)
            if data is not None:
                self.data.append(data)

    @staticmethod
    def _next_or_default_event(ev: Event) -> Event:
        if ev.next_event:
            return ev.next_event
        return ev

    @staticmethod
    def _unsqueeze_data(data: dict) -> Dict[str, Any]:
        new_data = {}
        for k, v in flatten(data).items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v).float()
            if v.dim() == 0:
                v = v.unsqueeze(0)
            elif v.dim() > 1:
                v = v.squeeze()
            new_data[k] = v
        return cast(typ=Dict[str, Any], val=unflatten(new_data))

    @staticmethod
    def _buyer_seller_price(ev: Event) -> Tuple[float, float]:
        buyer_price = ev.get_price(Agent.BUYER)
        seller_price = ev.get_price(Agent.SELLER)
        if buyer_price is None:
            buyer_price = 0
        if seller_price is None:
            seller_price = ev.scenario.list_price
        return buyer_price, seller_price

    @classmethod
    def _format_state(cls, ev: Event) -> dict:
        buyer_price, seller_price = cls._buyer_seller_price(ev)

        return {
            "utterance": ev.data["embeddings"]["dialog"],
            "action_type": event_to_int(ev.event),
            # HACK: some prices # are off the charts.. clip it to a constant
            "buyer_price": min(buyer_price / ev.scenario.list_price, 1.5),
            "seller_price": min(seller_price / ev.scenario.list_price, 1.5),
        }

    @classmethod
    def _format_action(cls, ev: Event) -> dict:
        _, seller_price = cls._buyer_seller_price(ev)
        return {
            "utterance": ev.data["embeddings"]["dialog"],
            "action_type": event_to_int(ev.event),
            "price": min(seller_price / ev.scenario.list_price, 3),
        }

    @staticmethod
    def _calc_done_type(ev: Event) -> DoneType:
        # event must be a seller type
        assert ev.agent == Agent.SELLER

        # check if seller caused end of episode
        if isinstance(ev.event, Quit) or isinstance(ev.event, Reject):
            return DoneType.SELLER_REJECT
        elif isinstance(ev.event, Accept):
            return DoneType.SELLER_ACCEPT

        # get all consecutive buyer events
        ev2: Optional[Event] = ev.next_event
        buyer_evs = []
        while ev2 is not None and ev2.agent == Agent.BUYER:
            buyer_evs.append(ev2)
            ev2 = ev2.next_event

        # check if buyer quit
        if any(
            [
                isinstance(ev.event, Quit) or isinstance(ev.event, Reject)
                for ev in buyer_evs
            ]
        ):
            return DoneType.BUYER_REJECT
        elif any([isinstance(ev.event, Accept) for ev in buyer_evs]):
            return DoneType.BUYER_ACCEPT

        return DoneType.NOT_DONE

    @staticmethod
    def _calc_sent_type(sent: str) -> int:
        sent = sent.strip()
        if sent == "accept":
            return event_to_int(Accept())
        elif sent == "reject":
            return event_to_int(Reject())
        elif sent.startswith("offer"):
            return event_to_int(Offer(0))
        elif sent == "quit":
            return event_to_int(Quit())
        return event_to_int(Message(sent))

    def _format_data(self, ev: Event) -> Optional[Dict[str, Any]]:
        """Attempts to format an event. If the event cannot be formatted into
        a valid seller (s,a,r,s) pair, then returns None. Else,
        returns the transition.

        """
        # must be a seller event that is not the first event
        if ev.agent != Agent.SELLER or ev.prev_event is None:
            return None

        # we will not assume which agent comes before or after the
        # current event.

        # rename the events
        obs_ev = ev.prev_event
        act_ev = ev
        next_obs_ev = self._next_or_default_event(ev)

        data = {
            "obs": self._format_state(obs_ev),
            "act": self._format_action(act_ev),
            "done": int(self._calc_done_type(ev) != DoneType.NOT_DONE),
            "rew": self.reward_handler.compute_reward(ev),
            "next_obs": self._format_state(next_obs_ev),
            # sentence options
            "cur_choices": obs_ev.data["embeddings"]["sentences"],
            "cur_choices_type": [
                self._calc_sent_type(s) for s in obs_ev.data["sentences"]
            ],
            "next_choices": next_obs_ev.data["embeddings"]["sentences"],
            "next_choices_type": [
                self._calc_sent_type(s) for s in next_obs_ev.data["sentences"]
            ],
        }
        return self._unsqueeze_data(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class RewardHandler(object):
    def compute_reward(self, ev: Event) -> float:
        raise NotImplementedError()


class UtilityReward(RewardHandler):
    REWARD_PRICE_DECREASE = -2
    REWARD_NO_DEAL = -20
    REWARD_SCALE = 10

    def __init__(self, price_decrease_penalty: bool):
        self.price_decrease_penalty = price_decrease_penalty

    def compute_reward(self, ev: Event) -> float:
        # we need to handle each done type individually
        done = CraigslistData._calc_done_type(ev)

        if done == DoneType.BUYER_REJECT or done == DoneType.SELLER_REJECT:
            # we failed to reach a deal
            return self.REWARD_NO_DEAL

        elif done == DoneType.BUYER_ACCEPT or done == DoneType.SELLER_ACCEPT:
            # we made a deal: evaluate how much money we earned.
            price = ev.get_price(Agent.SELLER)
            if price is None:
                # no price decided
                return self.REWARD_NO_DEAL
            # scale the price
            price /= ev.scenario.list_price
            if price < 0.5 or price > 1.5:
                # price too high or too low
                return self.REWARD_NO_DEAL
            # balance out price with negative reward
            return self.REWARD_SCALE * (price - 1)
        elif done == DoneType.NOT_DONE:
            # no deal yet, continue only if price decrease penalty
            if not self.price_decrease_penalty:
                return 0

            # did we agree on a price yet
            cur_seller_price = ev.get_price(Agent.SELLER)
            if cur_seller_price is None:
                return 0

            # check whether the seller price has decreased
            assert ev.prev_event is not None
            cur_price = ev.get_price(Agent.SELLER)
            prev_price = ev.prev_event.get_price(Agent.SELLER)
            if cur_price is None or prev_price is None:
                return 0
            if cur_price > prev_price:
                return self.REWARD_PRICE_DECREASE
            return 0
        else:
            raise NotImplementedError


class FairnessReward(RewardHandler):
    def compute_reward(self, ev: Event) -> float:
        # we need to handle each done type individually
        target_price = (ev.scenario.buyer_target + ev.scenario.seller_target) / 2.0
        cur_price = ev.get_price(Agent.SELLER)
        if cur_price is None:
            return 0.0
        else:
            return -np.abs(cur_price - target_price)


class AgreementReward(RewardHandler):
    def compute_reward(self, ev: Event) -> float:
        # we need to handle each done type individually
        done = CraigslistData._calc_done_type(ev)
        if done == DoneType.BUYER_REJECT or done == DoneType.SELLER_REJECT:
            return -1.0
        elif done == DoneType.BUYER_ACCEPT or done == DoneType.SELLER_ACCEPT:
            return 1.0
        return 0.0
