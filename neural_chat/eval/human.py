from typing import List
from .base import Model
import neural_chat.craigslist as cg


class Human(Model):
    @staticmethod
    def reset(_):
        pass

    @staticmethod
    def response(*_, **__) -> cg.EventType:
        utterance = input(">> ").strip()
        if utterance == "accept":
            return cg.Accept()
        elif utterance == "reject":
            return cg.Reject()
        elif utterance == "quit":
            return cg.Quit()
        elif utterance.startswith("offer"):
            return cg.Offer(float(utterance[len("offer") :].strip()))
        return cg.Message(utterance)
