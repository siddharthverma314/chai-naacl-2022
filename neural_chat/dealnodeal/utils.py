from typing import Iterable, List
import neural_chat.craigslist as cg
from neural_chat import utils
from pathlib import Path


def load_craigslist(split: str = "train") -> cg.Craigslist:
    return cg.Craigslist(str(Path(utils.COCOA_DATA_DIR) / f"{split}.json"))


def format_scenario(scenario: cg.Scenario) -> str:
    return str(scenario) + f"\nprice {scenario.list_price}" + "<eof>"


def format_event_pair(agent: cg.Agent, event: cg.EventType) -> str:
    if agent == cg.Agent.BUYER:
        out = "<buyer>"
    else:
        out = "<seller>"

    # add event
    if isinstance(event, cg.Offer):
        out += f"offer {event.price}"
    else:
        out += str(event)

    return out


def format_events(events: List[cg.Event]) -> str:
    out = ""
    for ev in events:
        out += format_event_pair(ev.agent, ev.event)
    out += "<eof>"
    return out


def iter_context() -> Iterable[str]:
    data = load_craigslist()
    return map(format_scenario, data.scenarios.values())


def iter_utterance() -> Iterable[str]:
    data = load_craigslist()
    for scene in data.scenes:
        yield format_events(scene.events)
