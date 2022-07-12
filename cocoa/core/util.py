import random
import json
import string
import pickle
import numpy as np
import uuid


def random_multinomial(probs):
    target = random.random()
    i = 0
    accum = 0
    while True:
        accum += probs[i]
        if accum >= target:
            return i
        i += 1


def generate_uuid(prefix):
    # return prefix + '_' + ''.join([random.choice(string.digits + string.letters) for _ in range(16)])
    return prefix + "_" + str(uuid.uuid4())


def read_json(path):
    return json.load(open(path))


def write_json(raw, path):
    with open(path, "w") as out:
        print >> out, json.dumps(raw)


def read_pickle(path, encoding="ASCII"):
    with open(path, "rb") as fin:
        return pickle.load(fin, encoding=encoding)


def write_pickle(obj, path):
    with open(path, "wb") as fout:
        pickle.dump(obj, fout)


def normalize(a):
    ma = np.max(a)
    mi = np.min(a)
    assert ma > mi
    a = (a - mi) / (ma - mi)
    return a
