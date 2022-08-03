import logging
import numpy as np
import os
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LibriSpeechDataset, collate_fn
from helpers import set_seeds
from model import AutomaticSpeechRecognition

TRAIN = "train"
VALID = "validation"
TEST = "test"

# Replace this with MLFlow
base_dir = os.path.dirname(os.path.abspath(__file__))
save_to = os.path.join(base_dir, "results")
if not os.path.exists(save_to):
    os.makedirs(save_to)


def run_epoch(phase, epoch, model, dataloader, optimizer, criterion, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = phase == TRAIN

    if training:
        model.train()
    else:
        model.eval()

    losses = []
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - {phase}")
    for _, melspecs, len_melspecs, transcripts, len_transcripts in progress_bar:
        melspecs = melspecs.to(device)
        len_melspecs = len_melspecs.to(device)
        transcripts = transcripts.to(device)
        len_transcripts = len_transcripts.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            outputs = model(melspecs, len_melspecs)
            loss = criterion(outputs, transcripts, len_melspecs, len_transcripts)

            if training:
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            progress_bar.set_postfix(loss=np.mean(losses))

    mean_loss = np.mean(losses)

    info = {
        "loss": mean_loss
    }

    return info


def run_test(phase, epoch, model, dataloader, optimizer, criterion, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    losses = []
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - {phase}")
    for _, melspecs, len_melspecs, transcripts, len_transcripts in progress_bar:
        melspecs = melspecs.to(device)
        len_melspecs = len_melspecs.to(device)
        transcripts = transcripts.to(device)
        len_transcripts = len_transcripts.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            outputs = model(melspecs, len_melspecs)
            loss = criterion(outputs, transcripts, len_melspecs, len_transcripts)
            losses.append(loss.item())
            progress_bar.set_postfix(loss=np.mean(losses))

    mean_loss = np.mean(losses)

    info = {
        "loss": mean_loss
    }

    return info


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on '{device.type}'")

    best_model_filepath = os.path.join(save_to, "best_model.pt")
    last_model_filepath = os.path.join(save_to, "last_model.pt")

    train_dataset = LibriSpeechDataset(cfg["train_dir"])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        worker_init_fn=set_seeds,
    )

    valid_dataset = LibriSpeechDataset(cfg["valid_dir"])
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        worker_init_fn=set_seeds,
    )

    test_dataset = LibriSpeechDataset(cfg["train_dir"])
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        worker_init_fn=set_seeds,
    )


    in_features = cfg["melspec_params"]["n_mels"]
    n_classes = len(train_dataset.vocabulary)
    model = AutomaticSpeechRecognition(in_features=in_features, out_features=n_classes)
    if cfg.get("state_dict_filepath") is not None:
        state_dict = torch.load(cfg.get("state_dict_filepath"), map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)

    loss_fn = nn.CTCLoss()
    optimizer = Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

    epochs = range(1, cfg["n_epochs"] + 1)
    best_metric = torch.inf
    epochs_since_best = 0

    for epoch in epochs:
        info_train = run_epoch(
            phase=TRAIN,
            epoch=epoch,
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=loss_fn,
            device=device
        )

        info_valid = run_epoch(
            phase=VALID,
            epoch=epoch,
            model=model,
            dataloader=valid_dataloader,
            optimizer=optimizer,
            criterion=loss_fn,
            device=device
        )

        scheduler.step(info_valid["loss"])

        if info_valid["loss"] <  best_metric:
            best_metric = info_valid["loss"]
            torch.save(model.state_dict(), best_model_filepath)
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        torch.save(model.state_dict(), last_model_filepath)

        if epochs_since_best > cfg["patience"]:
            break

    best_model = AutomaticSpeechRecognition(in_features=in_features, out_features=n_classes)
    state_dict = torch.load(best_model_filepath, map_location=device)
    best_model.load_state_dict(state_dict)
    best_model.to(device)

    info_test = run_test(
        phase=TEST,
        epoch=0,
        model=best_model,
        dataloader=test_dataloader,
        criterion=loss_fn,
        device=device
    )


if __name__ == "__main__":
    cfg = {
        "train_dir": "/home/vsouzari/Documents/datasets/LibriSpeech/train-clean-100",
        "valid_dir": "/home/vsouzari/Documents/datasets/LibriSpeech/dev-clean",
        "test_dir": "/home/vsouzari/Documents/datasets/LibriSpeech/test-clean",
        "melspec_params": {
            "sample_rate": 16000,
            "n_fft": 1024,
            "win_length": 1024,
            "hop_length": 256,
            "n_mels": 80,
            "f_min": 0,
            "f_max": None
        },
        "learning_rate": 0.0001,
        "weight_decay": 0.001,
        "batch_size": 2,
        "n_epochs": 1,
        "patience": 20
    }

    main(cfg)

