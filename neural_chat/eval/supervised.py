from typing import List
from .base import Model
import neural_chat.craigslist as cg
from neural_chat.gpt2 import GPT2


class Supervised(Model):
    def __init__(self, gpt_dir: str):
        self.gpt = GPT2(gpt_dir)
        self.dialog: List[cg.EventType] = []

    @staticmethod
    def _str_to_event(utterance: str) -> cg.EventType:
        if utterance == "no deal":
            return cg.Reject()
        elif utterance == "deal":
            return cg.Accept()
        elif utterance == "quit":
            return cg.Quit()
        elif "$OFFER" in utterance:
            price = float(utterance[len("$OFFER") :])
            return cg.Offer(price)
        return cg.Message(utterance)

    @staticmethod
    def _event_to_str(ev: cg.EventType) -> str:
        if isinstance(ev, cg.Offer):
            return "$OFFER" + str(ev.price)
        return str(ev)

    def reset(self, scenario: cg.Scenario) -> None:
        self.scenario = scenario
        self.dialog = []

    def response(self, ev: cg.EventType, debug: bool = False) -> cg.EventType:
        self.dialog.append(ev)
        gen = self.gpt.generate(
            self.scenario,
            [self._event_to_str(d) for d in self.dialog],
            num_outputs=1,
        )[0]
        if debug:
            print(gen)
        return self._str_to_event(gen)
