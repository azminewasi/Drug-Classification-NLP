#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
from itertools import product

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


def read_csv(
    csv_file,
    x_col="smiles",
    y_col="tags",
):
    df = pd.read_csv(csv_file)

    all_y = set()
    all_x = set()

    # drop multi columns
    df = df[~df[y_col].str.contains(" ")]

    x = df[x_col]
    y = df[y_col]

    # find all y
    for item_y in y:
        all_y.update(item_y.split(" "))

    # make y mapping
    mapping_y = {val: index for index, val in enumerate(sorted(list(all_y)))}

    # find all x
    for item_x in x:
        all_x.update(set(item_x))

    # make x mapping
    mapping_x = {val: index + 1 for index, val in enumerate(sorted(list(all_x)))}
    mapping_x["<pad>"] = 0

    # encode y
    ys = [mapping_y[i] for i in y]
    ys = np.array(ys)

    # encode x
    xs = []
    for item_x in x:
        encoded_item = [mapping_x[c] for c in item_x]
        xs.append(encoded_item)
    xs = [np.array(item) for item in xs]

    to_return = {
        "x": {"raw": x.values, "data": xs},
        "y": {"data": ys},
        "mapping": {"x": mapping_x, "y": mapping_y},
    }
    return to_return


def split_data(data, ratio_dev=0.1, ratio_test=0.1, seed=None):
    # random number generator
    rng = np.random.default_rng(seed=seed)

    # dataset sizes
    size_total = len(data["y"]["data"])
    ratios = {"dev": ratio_dev, "test": ratio_test}
    sizes = {}
    for split, ratio in ratios.items():
        sizes[split] = int(ratio * size_total)
    sizes["train"] = size_total - sum(sizes.values())

    # split
    index = np.arange(size_total)
    rng.shuffle(index)

    indices = {}
    start = 0
    for split, size in sizes.items():
        indices[split] = index[start : start + size]
        start += size

    splits = {}
    for split, index in indices.items():
        x_data = data["x"]
        x_data = {k: [v[i] for i in index] for k, v in x_data.items()}

        y_data = data["y"]
        y_data = {k: v[index] for k, v in y_data.items()}

        splits[split] = {"x": x_data, "y": y_data}

    return splits


def make_n_gram_mapping(mapping, n):
    values = mapping.keys()
    combos = product(values, repeat=n)
    mapping = {"".join(v): i for i, v in enumerate(sorted(combos))}
    return mapping


def count_n_grams(text, n):
    len_gram = len(text) + 1 - n
    n_grams = [text[i : i + n] for i in range(len_gram)]
    return Counter(n_grams)


def get_topk_n_grams(data, n, topk=1000):
    counters = [count_n_grams(text, n) for text in data]
    counter = Counter()
    for c in counters:
        counter += c
    results = [w for w, _ in counter.most_common(topk)]
    return results


def sequence_collate(batch):
    x, y = zip(*batch)
    x = [torch.LongTensor(item) for item in x]
    lens = torch.LongTensor([len(i) for i in x])
    x_padded = pad_sequence(x, batch_first=True, padding_value=0)
    y = torch.LongTensor(np.array(y))
    _, perm_idx = lens.sort(0, descending=True)
    return x_padded[perm_idx], y[perm_idx], lens[perm_idx]


class NgramDataset(Dataset):
    """
    Encoder based on n grams
    """

    def __init__(self, x, y, top_grams=None, n=1, topk=1000):
        data_x = x["raw"]
        data_y = y["data"]
        if top_grams is None:
            top_grams = get_topk_n_grams(data_x, n, topk=topk)

        all_grams = []
        for item_x in data_x:
            unk = 0  # other tokens
            grams = count_n_grams(item_x, n)
            item = [grams[g] for g in top_grams]
            unk = [v for k, v in grams.items() if k not in top_grams]  # unk
            unk = sum(unk)
            item.append(unk)
            all_grams.append(item)

        self.top_grams = top_grams
        self.x = np.array(all_grams, dtype="float32")
        self.x_raw = data_x
        self.y = np.array(data_y, dtype="long")

    def __getitem__(self, index):
        item_x = self.x[index]
        item_y = self.y[index]

        return item_x, item_y

    def __len__(self):
        return len(self.x)


class SequenceDataset(Dataset):
    """
    Encode each character in sequence.
    0: padding
    """

    def __init__(self, x, y, mapping_x, mapping_y, n=1):
        data_x = x["data"]
        data_y = y["data"]

        self.x = data_x

        self.x_raw = x["raw"]
        self.y = np.array(data_y, dtype="int64")

        self.mapping_x = mapping_x
        self.mapping_x_inverse = {v: k for k, v in self.mapping_x.items()}
        self.mapping_y = mapping_y
        self.mapping_y_inverse = {v: k for k, v in self.mapping_y.items()}

    def __getitem__(self, index):
        item_x = np.array(self.x[index], dtype="int64")
        item_y = self.y[index]

        return item_x, item_y

    def __len__(self):
        return len(self.x)


def get_loaders_n_gram(
    csv_file, n=1, topk=20, ratio_dev=0.1, ratio_test=0.1, batch_size=32, seed=None
):
    data = read_csv(csv_file)
    mapping_x = data["mapping"]["x"]
    mapping_y = data["mapping"]["y"]
    splits = split_data(
        data,
        ratio_dev=ratio_dev,
        ratio_test=ratio_test,
        seed=seed,
    )

    # make train sets
    split_train = splits.pop("train")
    dataset_train = NgramDataset(split_train["x"], split_train["y"], n=n, topk=topk)
    top_grams = dataset_train.top_grams

    datasets = {
        k: NgramDataset(v["x"], v["y"], n=n, top_grams=top_grams)
        for k, v in splits.items()
    }
    datasets["train"] = dataset_train
    # batch size * 2 for train
    batch_sizes = {
        k: batch_size if k == "train" else batch_size * 2 for k in datasets.keys()
    }
    # shuffle only the train set
    shuffle = {k: True if k == "train" else False for k in datasets.keys()}
    # make loaders
    loaders = {
        k: DataLoader(v, batch_size=batch_sizes[k], shuffle=shuffle[k])
        for k, v in datasets.items()
    }
    # find sizes
    size_x = len(top_grams) + 1
    size_y = len(mapping_y)
    return {"loaders": loaders, "sizes": {"x": size_x, "y": size_y}}


def get_loaders_sequence(
    csv_file,
    ratio_dev=0.1,
    ratio_test=0.1,
    batch_size=32,
    seed=None,
):
    data = read_csv(csv_file)
    mapping_x = data["mapping"]["x"]
    mapping_y = data["mapping"]["y"]
    splits = split_data(
        data,
        ratio_dev=ratio_dev,
        ratio_test=ratio_test,
        seed=seed,
    )

    datasets = {
        k: SequenceDataset(v["x"], v["y"], mapping_x, mapping_y)
        for k, v in splits.items()
    }
    # batch size * 2 for train
    batch_sizes = {
        k: batch_size if k == "train" else batch_size * 2 for k in datasets.keys()
    }
    # shuffle only the train set
    shuffle = {k: True if k == "train" else False for k in datasets.keys()}
    # make loaders
    loaders = {
        k: DataLoader(
            v,
            batch_size=batch_sizes[k],
            shuffle=shuffle[k],
            collate_fn=sequence_collate,
        )
        for k, v in datasets.items()
    }
    # find sizes
    size_x = len(mapping_x)
    size_y = len(mapping_y)
    return {"loaders": loaders, "sizes": {"x": size_x, "y": size_y}}
