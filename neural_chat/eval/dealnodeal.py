from typing import List, Tuple
from .base import Model
import neural_chat.craigslist as cg
from neural_chat.dealnodeal import DND, DNDTokenizer, format_scenario, format_event_pair
import itertools


class DealNoDeal(Model):
    def __init__(self, dnd_dir: str, agent: cg.Agent = cg.Agent.SELLER):
        self.dnd = DND.from_pretrained(dnd_dir)
        self.dnd_token = DNDTokenizer.from_pretrained(dnd_dir)
        self.dialog: List[cg.EventType] = []
        self.agent = agent

    @staticmethod
    def _str_to_event(utterance: str) -> cg.EventType:
        if utterance == "no deal":
            return cg.Reject()
        elif utterance == "deal":
            return cg.Accept()
        elif utterance == "quit":
            return cg.Quit()
        elif utterance.startswith("offer"):
            price = utterance[len("offer") :].strip().split(" ")[0]
            price = float(price)
            return cg.Offer(price)
        return cg.Message(utterance)

    @property
    def agent_dialog(self) -> str:
        agents = itertools.cycle([cg.Agent.SELLER, cg.Agent.BUYER])
        out = ""
        for agent, dialog in zip(agents, self.dialog):
            out += format_event_pair(agent, dialog)
        if self.agent == cg.Agent.SELLER:
            return out + "<seller>"
        else:
            return out + "<buyer>"

    def reset(self, scenario: cg.Scenario) -> None:
        self.scenario = scenario
        self.dialog = []

    def response(self, ev: cg.EventType, debug: bool = False) -> cg.EventType:
        self.dialog.append(ev)
        context = self.dnd_token(format_scenario(self.scenario), return_tensors="pt")
        utterance = self.dnd_token(self.agent_dialog, return_tensors="pt")
        if self.agent == cg.Agent.SELLER:
            eos_token_id = self.dnd_token.encode("<buyer>")[0]
        else:
            eos_token_id = self.dnd_token.encode("<seller>")[0]
        gen = self.dnd_token.decode(
            self.dnd.generate(
                input_ids=utterance.input_ids.to(self.dnd.device),
                max_length=512,
                context_input_ids=context.input_ids.to(self.dnd.device),
                use_cache=True,
                do_sample=True,
                num_beams=5,
                eos_token_id=eos_token_id,
                pad_token_id=self.dnd_token.pad_token_id,
            ).squeeze()
        )
        # process gen to remove random
        if self.agent == cg.Agent.SELLER:
            gen = gen.split("<seller>")[-1]
            gen = gen.split("<eof>")[0]
            gen = gen.split("<buyer>")[0]
        else:
            gen = gen.split("<buyer>")[-1]
            gen = gen.split("<eof>")[0]
            gen = gen.split("<seller>")[0]
        gen = gen.strip()
        if debug:
            print(gen)
        return self._str_to_event(gen)
