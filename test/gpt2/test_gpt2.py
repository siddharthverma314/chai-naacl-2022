from neural_chat.gpt2 import GPT2

SENTENCES = [
    "Selling a red car for $1000. In great shape.",
    "Hi! how many miles does the car have on it?",
    "Hi! Thank you for your interest. The car has only 1000 miles on it.",
    "Perfect. Will you be willing to take <price>?",
]

gpt2 = GPT2()


def test_gpt2_generation():
    sentences = gpt2.generate(SENTENCES[0], SENTENCES[1:])
    for sentence in sentences:
        print(sentence)
        assert "<sep>" not in sentence
        assert "<|endoftext|>" not in sentence


def test_gpt2_word_embedding():
    print(gpt2.embedding(SENTENCES).shape)
