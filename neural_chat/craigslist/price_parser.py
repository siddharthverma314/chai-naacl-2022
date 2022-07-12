from __future__ import annotations
from typing import List, Optional
from .types import *
from dataclasses import dataclass
import re

PRICE_REGEX = re.compile(r"(?P<dollar>\$?)(?P<num>\d([\d,]*\d)?(\.\d+)?)(?P<kilo>k?)")
LOWER_BOUND = 0.3
UPPER_BOUND = 3
TOLERANCE = 0.01


@dataclass
class Token:
    token: str
    price: Optional[float]


def parse_prices(
    list_price: float, text: str, lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND
) -> List[Token]:
    """Given a sentence, return a list of tokens with price
    annotations. Price must be between LOWER_BOUND*list_price and
    UPPER_BOUND*list_price to qualify.

    """

    tokens = []
    index = 0
    for match in PRICE_REGEX.finditer(text):
        # append token without price
        tokens.append(Token(token=text[index : match.start()], price=None))

        # calculate price
        price = float("".join([c for c in match.group("num") if c != ","]))
        if match.group("kilo") == "k":
            price *= 1000
        if (
            not match.group("dollar")
            and not lower_bound * list_price <= price <= upper_bound * list_price
        ):
            price = None

        # append price token
        tokens.append(Token(token=text[match.start() : match.end()], price=price))

        # move index forward
        index = match.span()[1]

    # final token
    tokens.append(Token(text[index:], price=None))
    return tokens


@dataclass
class PriceType:
    price: float
    ours: bool


@dataclass
class PriceData:
    our_price: Optional[float]
    their_price: Optional[float]
    utterance: str

    @staticmethod
    def from_agent(
        agent: Agent,
        buyer_price: Optional[float],
        seller_price: Optional[float],
        utterance: str,
    ) -> PriceData:
        if agent == Agent.BUYER:
            return PriceData(
                our_price=buyer_price, their_price=seller_price, utterance=utterance
            )
        elif agent == Agent.SELLER:
            return PriceData(
                our_price=seller_price, their_price=buyer_price, utterance=utterance
            )
        else:
            raise NotImplementedError

    def __str__(self):
        return self.utterance


@dataclass
class PriceParser:
    list_price: float
    buyer_price: Optional[float] = None
    seller_price: Optional[float] = None

    def get_price(self, agent: Agent) -> Optional[float]:
        if agent == Agent.BUYER:
            return self.buyer_price
        elif agent == Agent.SELLER:
            return self.seller_price
        raise NotImplementedError

    def set_price(self, agent: Agent, price: Optional[float]):
        if price:
            if price and agent == Agent.BUYER:
                self.buyer_price = price
            elif agent == Agent.SELLER:
                self.seller_price = price

    def _get_pricedata(self, agent: Agent, utterance: str):
        return PriceData(
            our_price=self.get_price(agent),
            their_price=self.get_price(agent.other_agent()),
            utterance=utterance,
        )

    def update_utterance(self, agent: Agent, utterance: str) -> PriceData:
        new_our_price = []
        new_their_price = []
        tokens = []

        their_price = self.get_price(agent.other_agent())

        # go through tokens to check who's price it is
        for tok in parse_prices(self.list_price, utterance):
            if their_price and tok.price and abs(tok.price - their_price) < TOLERANCE:
                new_their_price.append(tok.price)
                tokens.append("$PARTNER_PRICE")
            elif tok.price:
                new_our_price.append(tok.price)
                tokens.append("$PRICE")
            else:
                tokens.append(tok.token)

        # update currently set prices
        get_price_default = (
            lambda agent: self.get_price(agent) if self.get_price(agent) else 0
        )
        update = (
            lambda agent, prices: min(
                prices, key=lambda price: abs(price - get_price_default(agent))
            )
            if len(prices) > 0
            else None
        )

        new_our_price = update(agent, new_our_price)
        new_their_price = update(agent.other_agent(), new_their_price)

        if new_our_price:
            self.set_price(agent, new_our_price)
        if new_their_price:
            self.set_price(agent.other_agent(), new_their_price)

        return self._get_pricedata(agent, "".join(tokens))

    def update_event(self, event: EventType, agent: Agent) -> PriceData:
        if isinstance(event, Message):
            return self.update_utterance(agent, event.utterance)
        elif isinstance(event, Offer):
            self.set_price(agent, event.price)
            return self._get_pricedata(agent, str(event))
        elif isinstance(event, Accept):
            self.set_price(agent, self.get_price(agent.other_agent()))
            return self._get_pricedata(agent, str(event))
        else:
            return self._get_pricedata(agent, str(event))
