from neural_chat.craigslist import Craigslist
from pathlib import Path


def test_full():
    data_dir = Path(__file__).parent.parent.parent / "data"
    Craigslist(str(data_dir / "train.json"))
    Craigslist(str(data_dir / "test.json"))
