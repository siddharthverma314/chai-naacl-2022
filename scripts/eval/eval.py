from typing import Dict, Any, Optional
from neural_chat.eval import *

import neural_chat.craigslist as cg
import argparse
import random
import json


def rollout_2(
    buyer: Model,
    seller: Model,
    data: cg.Craigslist,
    debug: bool = False,
    scenario_id: Optional[str] = None,
) -> Dict[str, Any]:
    # sample random scenario
    if scenario_id is not None:
        scenario = data.scenarios[scenario_id]
    else:
        scenario = random.choice(list(data.scenarios.values()))
    debug_data = rollout(scenario, buyer, seller, debug, max_length=50)

    return {
        "buyer_price": debug_data.buyer_price,
        "seller_price": debug_data.seller_price,
        "scenario": scenario.scenario_id,
        "dialog": [Supervised._event_to_str(ev) for ev in debug_data.events],
        "accept": debug_data.accept,
    }


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--gpt-dir")
    parser.add_argument("--dnd-dir")
    parser.add_argument("--checkpoint-file")
    parser.add_argument("--buyer", default="human")
    parser.add_argument("--seller", default="ours")
    parser.add_argument("--num-outputs", type=int, default=5)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num-rollouts", type=int, default=30)
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()

    # load a random context
    data = cg.Craigslist(args.data_path)
    hack = False

    # setup agents
    buyer = None
    if args.buyer == "human":
        buyer = Human()
    elif args.buyer.startswith("theirs"):
        model_type = args.buyer.split("_")[1]
        buyer = Theirs(args.data_path, cg.Agent.BUYER, model_type, hack=hack)
    elif args.buyer == "supervised":
        buyer = Supervised(args.gpt_dir)
    assert buyer is not None, "Buyer type not implemented"

    seller = None
    if args.seller == "human":
        seller = Human()
    elif args.seller == "ours":
        seller = Ours(args.gpt_dir, args.checkpoint_file, args.num_outputs)
    elif args.seller.startswith("theirs"):
        model_type = args.seller.split("_")[1]
        seller = Theirs(args.data_path, cg.Agent.SELLER, model_type, hack=hack)
    elif args.seller == "supervised":
        seller = Supervised(args.gpt_dir)
    elif args.seller == "dnd":
        seller = DealNoDeal(args.dnd_dir)
    assert seller is not None, "Seller type not implemented"

    results = []
    i = 0
    while i < args.num_rollouts:
        try:
            results.append(
                rollout_2(
                    buyer,
                    seller,
                    data,
                    debug=args.debug,
                    scenario_id=args.scenario,
                )
            )
            print("ROLLOUT", i, results[-1])
            i += 1
        except Exception as e:
            print(e)
            continue

        with open(args.output_path, "w") as f:
            json.dump(results, f)
