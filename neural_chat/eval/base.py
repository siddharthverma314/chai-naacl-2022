from typing import Optional, List
import abc
import neural_chat.craigslist as cg
from dataclasses import dataclass


class Model(abc.ABC):
    """Abstract class that represents any agent that can respond to
    Craigslist events

    """

    @abc.abstractmethod
    def reset(self, scenario: cg.Scenario) -> None:
        pass

    @abc.abstractmethod
    def response(self, ev: cg.EventType, debug: bool = False) -> cg.EventType:
        pass


@dataclass
class RolloutData:
    buyer_price: float
    seller_price: float
    events: List[cg.EventType]
    agents: List[cg.Agent]
    accept: bool


def is_end(evs: List[cg.EventType]):
    if (
        isinstance(evs[-1], cg.Accept)
        or isinstance(evs[-1], cg.Reject)
        or isinstance(evs[-1], cg.Quit)
    ):
        return True
    elif (
        len(evs) > 2
        and isinstance(evs[-1], cg.Offer)
        and isinstance(evs[-2], cg.Offer)
        and evs[-1].price == evs[-2].price
    ):
        return True
    return False


def ev_str(ev: cg.EventType) -> str:
    if isinstance(ev, cg.Offer):
        return f"offer {ev.price}"
    return str(ev)


def rollout(
    scenario: cg.Scenario,
    buyer: Model,
    seller: Model,
    debug: bool = False,
    max_length: int = 15,
) -> RolloutData:
    # sample random scenario
    if debug:
        print("-" * 50)
        print("CONTEXT:", str(scenario))
        print("PRICE:", scenario.list_price)

    # initialize agents
    buyer.reset(scenario)
    seller.reset(scenario)
    models: List[Model] = [buyer, seller]
    dialog: List[cg.EventType] = [cg.Message("Hi!")]
    agents: List[cg.Agent] = [cg.Agent.BUYER]
    agent = cg.Agent.SELLER

    while not is_end(dialog) and len(dialog) < max_length:
        utterance = models[agent.value].response(dialog[-1], debug)
        dialog.append(utterance)
        agents.append(agent)

        if debug:
            print(f"{str(agent)}:", ev_str(utterance))

        agent = agent.other_agent()

    # figure out accept
    if isinstance(dialog[-1], cg.Accept):
        accept = True
    else:
        accept = False

    pp = cg.PriceParser(scenario.list_price, 0, scenario.list_price)
    agent = cg.Agent.BUYER
    for ev in dialog:
        pp.update_event(ev, agent)
        agent = agent.other_agent()

    return RolloutData(
        buyer_price=pp.buyer_price / pp.list_price,
        seller_price=pp.seller_price / pp.list_price,
        events=dialog,
        accept=accept,
        agents=agents,
    )
