from __future__ import annotations
from typing import Union
from dataclasses import dataclass
import enum


class Agent(enum.Enum):
    BUYER = 0
    SELLER = 1

    def other_agent(self) -> Agent:
        if self == self.BUYER:
            return self.SELLER
        elif self == self.SELLER:
            return self.BUYER
        else:
            raise NotImplementedError

    def __str__(self) -> str:
        if self == self.BUYER:
            return "buyer"
        elif self == self.SELLER:
            return "seller"
        else:
            raise NotImplementedError


###############
# event types #
###############


@dataclass
class Message:
    utterance: str

    def __str__(self):
        return self.utterance


@dataclass
class Offer:
    price: float

    def __str__(self):
        return "offer"


class Accept:
    @staticmethod
    def __str__() -> str:
        return "deal"


class Reject:
    @staticmethod
    def __str__() -> str:
        return "no deal"


class Quit:
    @staticmethod
    def __str__() -> str:
        return "quit"


EventType = Union[Message, Offer, Accept, Reject, Quit]


def event_to_int(ev: EventType) -> int:
    if isinstance(ev, Message):
        return 0
    elif isinstance(ev, Offer):
        return 1
    elif isinstance(ev, Accept):
        return 2
    elif isinstance(ev, Reject):
        return 3
    elif isinstance(ev, Quit):
        return 4
    else:
        raise NotImplementedError
