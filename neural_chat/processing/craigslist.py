import nltk
import collections
import re
from nltk.tokenize.destructive import NLTKWordTokenizer

PRICE_TOKEN = "<price>"
NUMBER_TOKEN = "<number>"

remove_tokens = set(
    [
        "?",
        ",",
        "\\",
        "/",
        "!",
        "$",
        "(",
        ")",
        "[",
        "]",
        "-",
        "&",
        ":",
        "''",
        "``",
        "<",
        ">",
        ";",
    ]
)

tokenizer = NLTKWordTokenizer()

year_re = re.compile(r"(20|19)[0-9][0-9]")
number_re = re.compile(r"[0-9]+(\.[0-9][0-9])?(\.)?$")

filter_price_words = set(
    [
        "year",
        "years",
        "month",
        "months",
        "day",
        "days",
        "minute",
        "minutes",
        "hour",
        "hours",
        "second",
        "seconds",
        "bed",
        "bath",
        "bdrm",
        "room",
        "piece",
        "pack",
        "long",
        "wide",
        "tall",
        "''",
        "%",
        "application",
        "by",
        "inches",
        "miles",
        "pounds",
        "meters",
        "pm",
        "am",
        "wheel",
        "or",
        "times",
    ]
)


def filter_price(
    pretoken, token, posttoken, list_price, min_multiple=0.3, max_multiple=3.0
):
    is_number = number_re.match(token)
    if not is_number:
        return token, None
    token = token.strip(".")
    number = float(token)
    in_range = (number >= (min_multiple * list_price)) and (
        number <= (max_multiple * list_price)
    )
    if posttoken in filter_price_words:
        return NUMBER_TOKEN, None
    elif pretoken == "$":
        return PRICE_TOKEN, number
    elif in_range:
        return PRICE_TOKEN, number
    else:
        return NUMBER_TOKEN, None


def process_str(sentence, list_price, min_multiple=0.3, max_multiple=3.0):
    """Tokenizes a string and replaces prices/numbers with special tokens."""
    sentence = sentence.lower()
    for remove_token in remove_tokens:
        sentence = sentence.replace(remove_token, "")
    tokenized = tokenizer.tokenize(sentence)
    N = len(tokenized)
    tokenized = [
        filter_price(
            tokenized[max(0, i - 1)],
            tokenized[i],
            tokenized[min(N - 1, i + 1)],
            list_price,
            min_multiple=min_multiple,
            max_multiple=max_multiple,
        )
        for i, token in enumerate(tokenized)
    ]
    if len(tokenized) == 0:
        return "", 0.0
    tokens, prices = zip(*tokenized)
    utterance = " ".join(tokens)
    prices = [price for price in prices if price is not None]
    if len(prices) > 1:
        # use the last price as a heuristic
        price = prices[-1]
    elif len(prices) == 1:
        price = prices[0]
    else:
        price = 0.0
    return utterance, price


def margin_reward(metadata, price):
    rewards = {}
    targets = {}
    kbs = metadata["kbs"]
    for agent_id in (0, 1):
        kb = kbs[agent_id]
        targets[kb["personal"]["Role"]] = kb["personal"]["Target"]

    midpoint = (targets["seller"] + targets["buyer"]) / 2.0
    norm_factor = abs(midpoint - targets["seller"])
    rewards["seller"] = (price - midpoint) / norm_factor
    # Zero sum
    rewards["buyer"] = -1.0 * rewards["seller"]
    return rewards
