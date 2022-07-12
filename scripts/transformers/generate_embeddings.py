from typing import Dict
from transformers.models.auto.tokenization_auto import AutoTokenizer
from neural_chat.craigslist.parse import Event
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from neural_chat.craigslist import Craigslist, Agent
from neural_chat.gpt2 import format_dialog
import argparse
import torch


class Worker:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.model = AutoModel.from_pretrained(self.checkpoint_dir).cuda()
        self.token = AutoTokenizer.from_pretrained(self.checkpoint_dir)

    def _embed(self, sentence: str) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.token(sentence + "<sep>", return_tensors="pt").to(
                self.model.device
            )
            embedding = self.model(**tokens)[0]
            attn = tokens["attention_mask"].unsqueeze(-1)
            return (attn * embedding).mean(axis=1).squeeze().detach().cpu()

    def process(self, ev: Event) -> Dict[str, torch.Tensor]:
        """Each embedding returned contains the information for the context +
        all dialog inclusive of the sentence, which replaces the last
        dialog if needed.

        """
        evs = ev.get_events()
        dialog = [(ev.agent, ev.data["price"].utterance) for ev in evs]
        return {
            "sentences": torch.stack(
                [
                    self._embed(
                        format_dialog(ev.scenario, dialog[:-1] + [(Agent.SELLER, s)])
                    )
                    for s in ev.data["sentences"]
                ]
            ),
            "dialog": self._embed(format_dialog(ev.scenario, dialog)),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir")
    parser.add_argument("--sentences", default="./sentences.pkl")
    parser.add_argument("--input-file", default="../../data/train.json")
    parser.add_argument("--output-file", default="embeddings.pkl")
    args = parser.parse_args()

    print("Opening input file...")
    cg = Craigslist(args.input_file, {"sentences": args.sentences})

    print("Loading worker...")
    worker = Worker(args.checkpoint_dir)

    print("Starting task...")
    data = {}
    for k, v in tqdm(cg.events.items()):
        data[k] = worker.process(v)

    print("Saving file")
    with open(args.output_file, "wb") as f:
        torch.save(data, f)
