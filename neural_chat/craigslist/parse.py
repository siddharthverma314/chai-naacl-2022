from __future__ import annotations
from neural_chat.craigslist.price_parser import PriceParser, PriceData
from typing import List, Optional, Dict, Any
from .types import *
from .price_parser import parse_prices
import json
import torch
import multiprocessing as mp


@dataclass
class Scenario:
    scenario_id: str
    title: str
    description: str
    list_price: float
    buyer_target: float
    seller_target: float

    @staticmethod
    def from_json(val: Any, scenario_id: str) -> Scenario:
        item = val["kbs"][0]["item"]
        list_price = item["Price"]

        # replace price in description
        description = "\n".join(item["Description"])
        description = "".join(
            [tok.token for tok in parse_prices(list_price, description, 0.9, 1.1)]
        )

        role_to_kb = {kb["personal"]["Role"]: kb for kb in val["kbs"]}

        return Scenario(
            scenario_id=scenario_id,
            title=item["Title"],
            description=description,
            list_price=list_price,
            buyer_target=role_to_kb["buyer"]["personal"]["Target"],
            seller_target=role_to_kb["seller"]["personal"]["Target"],
        )

    def __str__(self):
        return f"{self.title}\n{self.description}"


@dataclass
class Event:
    agent: Agent
    event: EventType
    event_id: str
    next_event: Optional[Event]
    prev_event: Optional[Event]
    scenario: Scenario
    data: Dict[str, Any]

    @staticmethod
    def from_json(val: Any, scenario: Scenario, scene_id: str, num: int) -> Event:
        if val["action"] == "message":
            event = Message(val["data"])
        elif val["action"] == "offer":
            event = Offer(val["data"]["price"])
        elif val["action"] == "accept":
            event = Accept()
        elif val["action"] == "reject":
            event = Reject()
        elif val["action"] == "quit":
            event = Quit()
        else:
            raise NotImplementedError

        return Event(
            agent=Agent(val["agent"]),
            event=event,
            event_id=f"{scene_id}_{num}",
            next_event=None,
            prev_event=None,
            scenario=scenario,
            data=dict(),
        )

    def get_events(self, direction="prev") -> List[Event]:
        if direction == "prev":
            func = lambda ev: ev.prev_event
        elif direction == "next":
            func = lambda ev: ev.next_event
        else:
            raise NotImplementedError

        events = []
        ev = self
        while ev is not None:
            events.append(ev)
            ev = func(ev)
        events.reverse()
        return events

    def get_price(self, agent: Agent) -> Optional[float]:
        pp: PriceData = self.data["price"]
        if agent == self.agent:
            return pp.our_price
        return pp.their_price


@dataclass
class Scene:
    scenario: Scenario
    events: List[Event]
    scene_id: str

    @staticmethod
    def from_json(val: Any) -> Scene:
        # parse data
        scenario = Scenario.from_json(val["scenario"], val["scenario_uuid"])
        scene_id = val["uuid"]
        events = [
            Event.from_json(ev, scenario, scene_id, i)
            for i, ev in enumerate(val["events"])
        ]

        # parse prices
        pp = PriceParser(scenario.list_price)
        for event in events:
            event.data["price"] = pp.update_event(event.event, event.agent)

        # link events
        for ev1, ev2 in zip(events, events[1:]):
            ev1.next_event = ev2
            ev2.prev_event = ev1

        return Scene(scenario, events, scene_id)


class Craigslist:
    def __init__(self, filepath: str, event_data: Dict[str, str] = {}):
        """Load a craigslist dataset, along with addition event_data. Event
        data is a mapping from name to filepath, and filepath is a
        pickle file that contains a mapping from event_id to data.
        """
        self.scenes = self.from_json(filepath)
        self.scenarios = {
            scene.scenario.scenario_id: scene.scenario for scene in self.scenes
        }
        self.events = {
            event.event_id: event for scene in self.scenes for event in scene.events
        }

        for name, pkl_path in event_data.items():
            with open(pkl_path, "rb") as f:
                data = torch.load(f)
            self.add_event_data(name, data)

    def add_event_data(self, name: str, event_data: Dict[str, Any]):
        for event_id, data in event_data.items():
            self.events[event_id].data[name] = data

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, i):
        return self.scenes[i]

    @staticmethod
    def from_json(filepath: str) -> List[Scene]:
        with open(filepath, "r") as f:
            data = json.load(f)

        assert isinstance(data, list)
        with mp.Pool() as pool:
            return pool.map(Scene.from_json, data)
