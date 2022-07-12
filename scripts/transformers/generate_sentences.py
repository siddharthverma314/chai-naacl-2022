from tqdm import tqdm
from neural_chat.gpt2 import GPT2
from neural_chat.craigslist import Craigslist
import argparse
import torch


parser = argparse.ArgumentParser()
parser.add_argument("checkpoint_dir")
parser.add_argument("--input-file", default="../../data/train.json")
parser.add_argument("--output-file", default="sentences.pkl")
parser.add_argument("--num-outputs", type=int, default=5)
args = parser.parse_args()

print("Opening input file...")
cg = Craigslist(args.input_file)

print("Loading GPT...")
gpt = GPT2(args.checkpoint_dir, device="cuda")
events = list(cg.events.values())
data = {}
for ev in tqdm(events):
    evs = ev.get_events()
    dialog = [ev.data["price"].utterance for ev in evs]
    agents = [ev.agent for ev in evs]
    val = gpt.generate(
        ev.scenario, dialog[:-1], agents[:-1], num_outputs=args.num_outputs
    )
    data[ev.event_id] = val

print("Saving file")
with open(args.output_file, "wb") as f:
    torch.save(data, f)
