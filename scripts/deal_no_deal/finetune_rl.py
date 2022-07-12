from typing import List
import argparse
import random
from neural_chat import utils
from neural_chat.dealnodeal import (
    DNDTokenizer,
    format_event_pair,
    load_craigslist,
    format_scenario,
)
from neural_chat.eval import Theirs, DealNoDeal, Model, rollout
import neural_chat.craigslist as cg
from transformers import PreTrainedTokenizerFast, DataCollatorWithPadding
import torch.optim
from tqdm import tqdm


class DNDDataCollator:
    def __init__(self, token: PreTrainedTokenizerFast, device="cuda"):
        self.pad = DataCollatorWithPadding(token, padding="longest")
        self.device = device

    def __call__(self, features: List[dict]):
        contexts = [f["context"] for f in features]
        utterances = [f["utterance"] for f in features]
        rewards = [f["reward"] for f in features]

        out = {}
        for k, v in self.pad(contexts).items():
            out[f"context_{k}"] = v
        for k, v in self.pad(utterances).items():
            out[f"utterance_{k}"] = v
        out["reward"] = torch.tensor(rewards)

        # move to device
        out = {k: v.to(self.device) for k, v in out.items()}

        return out


class DataCollector:
    def __init__(
        self,
        data: cg.Craigslist,
        token: PreTrainedTokenizerFast,
        buyer_model: Model,
        seller_model: Model,
        debug: bool = False,
    ):
        self.scenes = data.scenes
        self.token = token
        self.buyer_model = buyer_model
        self.seller_model = seller_model
        self.debug = debug

    def __len__(self):
        return len(self.scenes)

    @staticmethod
    def make_events(
        scenario: cg.Scenario, agents: List[cg.Agent], ev: List[cg.EventType]
    ) -> List[cg.Event]:
        events = [
            cg.Event(
                scenario=scenario,
                agent=a,
                event=e,
                event_id="",
                prev_event=None,
                next_event=None,
                data={},
            )
            for a, e in zip(agents, ev)
        ]
        for ev1, ev2 in zip(events, events[1:]):
            ev1.next_event = ev2
            ev2.prev_event = ev1

        pp = cg.PriceParser(list_price=scenario.list_price)
        for ev in events:
            ev.data["price"] = pp.update_event(ev.event, ev.agent)

        return events

    def collect(self, num: int = 1):
        scene = random.choice(self.scenes)
        data = []
        for _ in range(num):
            context = format_scenario(scene.scenario)
            events = rollout(
                scenario=scene.scenario,
                buyer=self.buyer_model,
                seller=self.seller_model,
                debug=self.debug,
            )
            if (
                len(events.events) > 2
                and isinstance(events.events[-1], cg.Offer)
                and isinstance(events.events[-2], cg.Offer)
                and events.events[-1].price == events.events[-2].price
            ):
                events.events[-1] = cg.Accept()
            utterance = (
                "".join(
                    [
                        format_event_pair(ag, ev)
                        for ev, ag in zip(events.events, events.agents)
                    ]
                )
                + "<eof>"
            )

            context = self.token(context)
            utterance = self.token(utterance)
            reward_calc = cg.UtilityReward(False)
            rewards = [
                reward_calc.compute_reward(ev)
                for ev in self.make_events(scene.scenario, events.agents, events.events)
                if ev.agent == cg.Agent.SELLER
            ]

            data.append(
                {
                    "context": context,
                    "utterance": utterance,
                    "reward": sum(rewards),
                    "context_hr": self.token.decode(context.input_ids),
                    "utterance_hr": self.token.decode(utterance.input_ids),
                }
            )
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-e", "--epochs", type=int, default=1000, help="Total epochs")
    parser.add_argument("-l", "--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-s", "--save_steps", type=int, default=20)
    parser.add_argument("--use-cocoa", action="store_true")
    parser.add_argument("--eval_steps", type=int, default=5)
    args = parser.parse_args()

    # model and data collection
    if args.use_cocoa:
        buyer_model = Theirs(
            utils.COCOA_DATA_DIR + "/train.json", cg.Agent.BUYER, hack=False
        )
    else:
        buyer_model = DealNoDeal(args.checkpoint, agent=cg.Agent.BUYER)
    seller_model = DealNoDeal(args.checkpoint)
    token: DNDTokenizer = DNDTokenizer.from_pretrained(args.checkpoint)
    dnd = seller_model.dnd
    dnd.train()
    dnd.cuda()
    optim = torch.optim.Adam(seller_model.dnd.parameters(), args.lr)

    # data
    train_data = load_craigslist("train")
    val_data = load_craigslist("dev")
    train_data = DataCollector(train_data, token, buyer_model, seller_model, args.debug)
    val_data = DataCollector(val_data, token, buyer_model, seller_model, args.debug)
    collate = DNDDataCollator(token)

    for epoch in tqdm(list(range(args.epochs))):
        data = collate(train_data.collect(5))
        nll = seller_model.dnd(**data).nll_seller
        rew = data["reward"] / 10
        loss = (nll * rew).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

        # log
        print(
            {
                "loss": loss.item(),
                "reward_mean": rew.mean().item(),
                "reward_min": rew.min().item(),
                "reward_max": rew.max().item(),
            }
        )

        # save
        if (epoch + 1) % args.save_steps == 0:
            path = args.output + f"/checkpoint_{epoch + 1}"
            dnd.save_pretrained(path)
            token.save_pretrained(path)
