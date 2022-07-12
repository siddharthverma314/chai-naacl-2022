from typing import List
import argparse
from neural_chat.dealnodeal import (
    DND,
    DNDConfig,
    DNDTokenizer,
    format_events,
    load_craigslist,
    format_scenario,
)
import neural_chat.craigslist as cg
from torch.utils.data import dataset
from transformers import PreTrainedTokenizerFast, trainer, DataCollatorWithPadding


class DNDDataCollator:
    def __init__(self, token: PreTrainedTokenizerFast):
        self.pad = DataCollatorWithPadding(token, padding="longest")

    def __call__(self, features: List[dict]):
        contexts = [f["context"] for f in features]
        utterances = [f["utterance"] for f in features]

        out = {}
        for k, v in self.pad(contexts).items():
            out[f"context_{k}"] = v
        for k, v in self.pad(utterances).items():
            out[f"utterance_{k}"] = v
        return out


class DNDDataset(dataset.Dataset):
    def __init__(self, data: cg.Craigslist, token: PreTrainedTokenizerFast):
        self.scenes = data.scenes
        self.token = token

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, i):
        scene = self.scenes[i]
        context = format_scenario(scene.scenario)
        utterance = format_events(scene.events)

        context = self.token(context)
        utterance = self.token(utterance)

        return {"context": context, "utterance": utterance}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument(
        "-e", "--epochs", type=int, default=20, help="Total number of epochs"
    )
    parser.add_argument("-l", "--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "-s",
        "--save_steps",
        type=int,
        default=500,
        help="How often to checkpoint model",
    )
    args = parser.parse_args()

    train_data = load_craigslist("train")
    val_data = load_craigslist("dev")
    token = DNDTokenizer.train_tokenizer()
    model = DND(
        DNDConfig(
            model_dim=256,
            token_dim=len(token),
            buyer_tok_id=token.encode("<buyer>")[0],
            seller_tok_id=token.encode("<seller>")[0],
        )
    )

    t = trainer.Trainer(
        model,
        data_collator=DNDDataCollator(token),
        train_dataset=DNDDataset(train_data, token),
        eval_dataset=DNDDataset(val_data, token),
        args=trainer.TrainingArguments(
            args.output,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            save_steps=args.save_steps,
        ),
        tokenizer=token,
    )
    t.train()
