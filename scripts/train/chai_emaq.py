#!/usr/bin/env python3

from flatten_dict.flatten_dict import flatten, unflatten
from neural_chat.algo import EMAQ_NOA
from neural_chat.actor import CraigslistDummyActor
from neural_chat.critic import DoubleQCritic
from neural_chat.logger import logger, Hyperparams
import neural_chat.craigslist as cg

import argparse
from torch.utils import data
from tqdm import tqdm

# move to device
def to(batch: dict, device):
    return unflatten({k: v.to(device) for k, v in flatten(batch).items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument("--filepath", type=str, required=True)
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--sentences", type=str, required=True)
    parser.add_argument("--path-length", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=str, default="256,256")
    parser.add_argument("--clip-log-prob", type=float, default=-10.0)
    parser.add_argument("--cql-weight", type=float, default=0.1)
    parser.add_argument("--price-decrease-penalty", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    args.hidden_dim = [int(d) for d in args.hidden_dim.split(",")]

    # data
    cdata = cg.CraigslistData(
        path=args.filepath,
        embeddings_path=args.embeddings,
        sentences_path=args.sentences,
        price_decrease_penalty=args.price_decrease_penalty,
    )
    data_loader = data.dataloader.DataLoader(
        dataset=cdata,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # models
    crt = DoubleQCritic(cdata.obs_spec, cdata.act_spec, hidden_dim=args.hidden_dim)
    act = CraigslistDummyActor(q_fn=crt)

    algo = EMAQ_NOA(
        action_space=cdata.act_spec,
        actor=act,
        critic=crt,
        cql_minq_weight=args.cql_weight,
        _device=args.device,
        _init_temperature=0.1,
    )

    # logging
    logger.initialize(
        {
            "algo": algo,
            "args": Hyperparams(vars(args)),
        },
        args.logdir,
    )
    logger.log_hyperparameters()

    # train
    for i in tqdm(list(range(args.num_epochs))):
        for j, sample in enumerate(tqdm(data_loader)):
            sample = to(sample, args.device)
            algo.update(sample, j)
        logger.epoch(i)
