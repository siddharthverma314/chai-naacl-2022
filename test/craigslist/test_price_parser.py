from neural_chat.craigslist import *


def test_parser():
    dialog1 = [
        "how about we settle on $750?",
        "how about 2,300?",
        "how about 2.3k?",
        "$2300 is too low, how about $2,200?",
        "I give you the extremely low price of 23.34",
        "No price here",
    ]

    prices = [
        [Token("$750", 750.0)],
        [Token("2,300", 2300.0)],
        [Token("2.3k", 2300.0)],
        [Token("$2300", 2300.0), Token("$2,200", 2200.0)],
    ]
    for d, ps in zip(dialog1, prices):
        assert [t for t in parse_prices(1000, d) if t.price != None] == ps


def test_price_parser():
    dialog = [
        "Hi! How are you doing",
        "I'm doing good, how are you",
        "Good, I was looking to buy the house for 900",
        "900 is too low, how about 950",
        "Let's meet in the middle and say 925",
        "deal. I'll come to pick it up at 3?",
        "Yes, sounds good!",
    ]
    outputs = [
        (None, None, "Hi! How are you doing"),
        (None, None, "I'm doing good, how are you"),
        (900.0, None, "Good, I was looking to buy the house for $PRICE"),
        (900.0, 950.0, "$PARTNER_PRICE is too low, how about $PRICE"),
        (925.0, 950.0, "Let's meet in the middle and say $PRICE"),
        (925.0, 950.0, "deal. I'll come to pick it up at 3?"),
        (925.0, 950.0, "Yes, sounds good!"),
    ]
    pp = PriceParser(1000)
    agent = Agent.BUYER
    for d, out in zip(dialog, outputs):
        data = pp.update_event(agent=agent, event=Message(d))
        assert data == PriceData.from_agent(agent, *out)
        agent = agent.other_agent()
    assert pp.update_event(agent=Agent.BUYER, event=Offer(500)) == PriceData(
        500.0, 950.0, "offer"
    )
    assert pp.update_event(agent=Agent.SELLER, event=Accept()) == PriceData(
        500.0, 500.0, "deal"
    )


def test_price_parser_2():
    dialog = [
        "hello",
        "how about $40?",
        "how about $1.00?",
        "no lets do $35.00",
        "deal",
        "i offer $25",
    ]
    outputs = [
        (None, None, "hello"),
        (None, 40.0, "how about $PRICE?"),
        (1.0, 40.0, "how about $PRICE?"),
        (1.0, 35.0, "no lets do $PRICE"),
        (1.0, 35.0, "deal"),
        (1.0, 25.0, "i offer $PRICE"),
    ]
    pp = PriceParser(list_price=50.0)
    agent = Agent.BUYER
    for d, out in zip(dialog, outputs):
        data = pp.update_event(agent=agent, event=Message(d))
        assert data == PriceData.from_agent(agent, *out)
        agent = agent.other_agent()
