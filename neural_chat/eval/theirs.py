from cocoa.sessions.session import Session
from neural_chat import utils
from cocoa.core.schema import Schema
from cocoa.craigslist.core.event import Event
from cocoa.craigslist.systems.neural_system import PytorchNeuralSystem
from cocoa.craigslist.core.price_tracker import PriceTracker
from cocoa.craigslist.systems.hybrid_system import HybridSystem
from cocoa.craigslist.model.generator import Templates, Generator
from cocoa.craigslist.core.kb import KB
from .base import Model
import neural_chat.craigslist as cg
import json
from pathlib import Path

# rulebased stuff
from cocoa.craigslist.systems.rulebased_system import RulebasedSystem
from cocoa.craigslist.model.manager import Manager

data_dir = Path(utils.COCOA_DATA_DIR)
schema_path = data_dir / "schema.json"
scenarios_path = data_dir / "dev.json"
price_tracker_path = data_dir / "price_tracker.pkl"
vocab_path = data_dir / "vocab.pkl"
templates_path = data_dir / "templates.pkl"
data_path = data_dir / "parsed.json"

schema = Schema(schema_path)

args = {
    "model": "lf2lf",
    "word_vec_size": 300,
    "dropout": 0.0,
    "encoder_type": "rnn",
    "decoder_type": "rnn",
    "context_embedder_type": "mean",
    "global_attention": "multibank_general",
    "share_embeddings": False,
    "share_decoder_embeddings": False,
    "enc_layers": 1,
    "copy_attn": False,
    "dec_layers": 1,
    "pretrained_wordvec": "",
    "rnn_size": 300,
    "rnn_type": "LSTM",
    "enc_layers": 1,
    "num_context": 2,
    "stateful": True,
    "sample": True,
    "max_length": 10,
    "n_best": 1,
    "batch_size": 128,
    "optim": "adagrad",
    "alpha": 0.01,
    "temperature": 0.5,
    "epochs": 30,
    "report_every": 500,
    "_vocab_pickle_path": vocab_path,
    "gpuid": [0],
}

# HACK: convert args from dict into object. Ex. args["epochs"]
# becomes args.epochs
args = type("args", (), args)


def kbs_map(data_path, agent):
    with open(data_path, "rb") as f:
        data = json.load(f)
    return {k["scenario_uuid"]: k["scenario"]["kbs"][agent] for k in data}


class Theirs(Model):
    def __init__(
        self,
        data_path,
        agent: cg.Agent = cg.Agent.SELLER,
        model_type: str = "utility",
        hack: bool = False,
    ):
        self.data_path = data_path
        self.model_type = model_type
        self.agent = agent.value
        self.hack = hack

    @staticmethod
    def _event_to_cocoa(agent: int, ev: cg.EventType) -> Event:
        if isinstance(ev, cg.Message):
            return Event(agent, None, "message", ev.utterance)
        elif isinstance(ev, cg.Accept):
            return Event(agent, None, "accept", None)
        elif isinstance(ev, cg.Reject):
            return Event(agent, None, "reject", None)
        elif isinstance(ev, cg.Quit):
            return Event(agent, None, "quit", None)
        return Event(agent, None, "offer", {"price": ev.price})

    def _cocoa_to_event(self, ev: Event) -> cg.EventType:
        if ev.action == "message":
            m: str = ev.data
            m = m.replace("$PRICE=", "")
            m = m.replace("$LIST_PRICE=", "")
            m = m.replace("$PARTNER_PRICE=", "")
            m = m.replace("$MY_PRICE=", "")
            return cg.Message(m)
        elif ev.action == "accept":
            return cg.Accept()
        elif ev.action == "reject":
            return cg.Reject()
        elif ev.action == "quit":
            return cg.Quit()
        elif ev.action == "offer":
            return cg.Offer(ev.data["price"])
        raise NotImplementedError

    def reset(self, scenario: cg.Scenario) -> None:
        price_tracker = PriceTracker(price_tracker_path)
        templates = Templates.from_pickle(templates_path)
        generator = Generator(templates)
        if self.model_type == "rulebased":
            with open(data_path) as f:
                data = json.load(f)
            intents = []
            for d in data:
                foo = ["<start>"]
                for ev in d["events"]:
                    if ev["metadata"]:
                        foo.append(ev["metadata"]["intent"])
                intents.append(foo)
            manager = Manager.from_train(intents)
            system = RulebasedSystem(
                price_tracker, generator, manager, timed_session=False
            )
        else:
            mp = data_dir / f"model_{self.model_type}.pt"
            system = PytorchNeuralSystem(args, schema, price_tracker, mp, timed=False)
            if not self.hack:
                system = HybridSystem(
                    price_tracker, generator, system, timed_session=False
                )

        # make sure description is not empty
        kb_json = kbs_map(self.data_path, self.agent)[scenario.scenario_id]
        desc = kb_json["item"]["Description"]
        if len(desc):
            desc.append("empty")

        self.session: Session = system.new_session(
            self.agent,
            KB.from_dict(schema.attributes, kb_json),
        )

    def response(self, ev: cg.EventType, _: bool = False) -> cg.EventType:
        self.session.receive(self._event_to_cocoa(1 - self.agent, ev))
        ev2 = self._cocoa_to_event(self.session.send())
        if (
            isinstance(ev, cg.Offer)
            and isinstance(ev2, cg.Offer)
            and ev.price == ev2.price
        ):
            return cg.Accept()
        return ev2
