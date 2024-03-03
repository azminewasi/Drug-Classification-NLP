#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from data import get_loaders_n_gram
from methods import MLP


def train(loader_train, loader_dev, model, device, optimizer, n_epochs):
    acc_best = 0
    model_best = None
    criterion = nn.CrossEntropyLoss()

    bar_epochs = tqdm(range(n_epochs), leave=False)
    for epoch in bar_epochs:
        # train
        bar_epoch = tqdm(loader_train, disable=True, leave=False)
        model.train()
        for x, y in bar_epoch:
            x = x.to(device)
            y = y.to(device)
            y_out = model(x)
            loss = criterion(y_out, y.type(torch.LongTensor))
            loss.backward()
            optimizer.step()
            loss_iter = loss.item()
            bar_epoch.set_postfix({"loss": loss_iter})
        bar_epoch.close()

        bar_dev = tqdm(loader_dev, disable=True, leave=False)
        model.eval()

        # val
        ys_pred, ys_true = [], []
        with torch.no_grad():
            for x, y in bar_dev:
                x = x.to(device)
                y = y.to(device)
                y_out = model(x)
                y_pred = torch.argmax(y_out, axis=1)
                ys_pred.append(y_pred.cpu())
                ys_true.append(y.cpu())
        bar_dev.close()
        ys_pred = torch.cat(ys_pred)
        ys_true = torch.cat(ys_true)
        acc = (ys_pred == ys_true).float().mean()
        acc = acc.item() * 100
        if acc > acc_best:
            acc_best = acc
            model_best = copy.deepcopy(model)
        bar_epochs.set_postfix({"acc_best": acc_best})

    return model_best


def test(loader_test, model, device):
    model.eval()
    ys_pred, ys_true = [], []
    bar_test = tqdm(loader_test, leave=False)
    with torch.no_grad():
        for x, y in bar_test:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            y_pred = torch.argmax(y_pred, axis=1)
            ys_pred.append(y_pred.cpu())
            ys_true.append(y.cpu())

    bar_test.close()

    ys_pred = torch.cat(ys_pred)
    ys_true = torch.cat(ys_true)

    return ys_pred, ys_true


def run(
    csv_file,
    seed,
    n=5,
    topk=1000,
    ratio_dev=0.1,
    ratio_test=0.1,
    batch_size=32,
    size_hidden=None,
    dropout=0.1,
    n_epochs=50,
    lr=3e-4,
    weight_decay=0,
):
    # data settings
    ratio_dev = ratio_dev
    ratio_test = ratio_test
    batch_size = batch_size
    n = n
    data = get_loaders_n_gram(
        csv_file,
        n=n,
        topk=topk,
        ratio_dev=ratio_dev,
        ratio_test=ratio_test,
        seed=seed,
        batch_size=batch_size,
    )
    size_x = data["sizes"]["x"]
    size_y = data["sizes"]["y"]
    loader_train = data["loaders"]["train"]
    loader_dev = data["loaders"]["dev"]
    loader_test = data["loaders"]["test"]
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model settings
    if size_hidden is None:
        size_hidden = [size_x // 2, size_x // 4]
    size_hidden = [size_x] + size_hidden
    dropout = dropout
    model = MLP(
        size_in=size_x,
        size_out=size_y,
        size_hidden=size_hidden,
        dropout=dropout,
    )
    model = model.to(device)

    # training settings
    n_epochs = n_epochs
    lr = lr
    weight_decay = weight_decay
    optimizer = Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # train
    model_best = train(loader_train, loader_dev, model, device, optimizer, n_epochs)
    return test(loader_test, model_best, device)


if __name__ == "__main__":
    # data dir
    csv_file = "./_DATA/all_chem_df.csv"
    # number of trials
    n_trials = 5
    seeds = list(range(n_trials))
    # data settings
    topk = 1000
    ratio_dev = 0.1
    ratio_test = 0.2
    batch_size = 32
    # model settings
    n = 5
    dropout = 0.1
    size_hidden = [512, 256, 128, 32]
    # training settings
    n_epochs = 200
    lr = 3e-5
    weight_decay = 0

    

    for seed in seeds:
        y_pred, y_true = run(
            csv_file,
            seed,
            n,
            topk,
            ratio_dev,
            ratio_test,
            batch_size,
            size_hidden,
            dropout,
            n_epochs,
            lr,
        )
        log_file = f"./scores/MLP/{seed}-seed--{n}-gram--topk-{topk}--lr-{lr}.csv"
        with open(log_file, "a") as f:
            f.write("pred,true\n")
            for p, t in zip(y_pred, y_true):
                f.write(f"{p},{t}\n")
