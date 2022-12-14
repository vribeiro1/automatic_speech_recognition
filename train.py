import argparse
import logging
import mlflow
import numpy as np
import os
import tempfile
import torch
import torch.nn as nn
import yaml

from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchaudio.functional import edit_distance
from tqdm import tqdm

from dataset import LibriSpeechDataset, collate_fn
from decoder import GreedyCTCDecoder
from helpers import set_seeds
from model import AutomaticSpeechRecognition
from text_preprocessing import SILENCE

TRAIN = "train"
VALID = "validation"
TEST = "test"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

TMP_DIR = tempfile.mkdtemp(dir=RESULTS_DIR)


def run_epoch(phase, epoch, model, dataloader, optimizer, criterion, scheduler=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = phase == TRAIN

    if training:
        model.train()
    else:
        model.eval()

    losses = []
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - {phase}")
    for i, (_, melspecs, len_melspecs, transcripts, len_transcripts) in enumerate(progress_bar, start=epoch * int(len(dataloader))):
        melspecs = melspecs.to(device)
        transcripts = transcripts.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            outputs = model(melspecs, len_melspecs)
            loss = criterion(outputs, transcripts, len_melspecs, len_transcripts)

            if training:
                loss.backward()
                optimizer.step()

                if scheduler:
                    scheduler.step()

            losses.append(loss.item())

        mean_loss = np.mean(losses)
        progress_bar.set_postfix(loss=mean_loss)

    mean_loss = np.mean(losses)
    mlflow.log_metrics({
        f"{phase}_loss": mean_loss
    })


    info = {
        "loss": mean_loss
    }

    return info


def evaluate_results(emissions, targets, len_targets, labels, blank=0, reduction="mean", norm=True):
    if reduction == "none":
        fn_reduction = lambda x: x
    else:
        fn_reduction = getattr(torch, reduction)

    decoder = GreedyCTCDecoder(labels, blank=blank)
    decoded_transcripts = decoder(emissions)

    target_transcripts = []
    for indices, length in zip(targets, len_targets):
        tokens = [labels[i.item()] for i in indices[:length]]
        transcript = "".join(tokens)
        target_transcripts.append(transcript)

    wer = []
    cer = []
    for decoded_transcript, target_transcript in zip(decoded_transcripts, target_transcripts):
        wer_sentence = edit_distance(decoded_transcript.split(), target_transcript.split())
        cer_sentence = edit_distance(decoded_transcript, target_transcript)

        if norm:
            wer_sentence = wer_sentence / len(target_transcript.split())
            cer_sentence = cer_sentence / len(target_transcript)

        wer.append(wer_sentence)
        cer.append(cer_sentence)

    wer = fn_reduction(torch.tensor(wer, dtype=torch.float))
    cer = fn_reduction(torch.tensor(cer, dtype=torch.float))

    return wer, cer


def run_test(phase, epoch, model, dataloader, criterion, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    wer = []
    cer = []
    losses = []
    labels = dataloader.dataset.vocabulary_transposed
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - {phase}")
    for _, melspecs, len_melspecs, transcripts, len_transcripts in progress_bar:
        melspecs = melspecs.to(device)
        transcripts = transcripts.to(device)

        with torch.set_grad_enabled(False):
            emissions = model(melspecs, len_melspecs)
            loss = criterion(emissions, transcripts, len_melspecs, len_transcripts)
            losses.append(loss.item())
            progress_bar.set_postfix(loss=np.mean(losses))

        batch_wer, batch_cer = evaluate_results(emissions, transcripts, len_transcripts, labels)
        wer.append(batch_wer)
        cer.append(batch_cer)

    mean_loss = np.mean(losses)
    mean_wer = np.mean(wer)
    mean_cer = np.mean(cer)

    info = {
        "loss": mean_loss,
        "wer": mean_wer,
        "cer": mean_cer
    }

    return info


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on '{device.type}'")

    best_model_filepath = os.path.join(TMP_DIR, "best_model.pt")
    last_model_filepath = os.path.join(TMP_DIR, "last_model.pt")

    train_dataset = LibriSpeechDataset(cfg["train_dir"], **cfg["melspec_params"])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        worker_init_fn=set_seeds,
    )

    valid_dataset = LibriSpeechDataset(cfg["valid_dir"], **cfg["melspec_params"])
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        worker_init_fn=set_seeds,
    )

    test_dataset = LibriSpeechDataset(cfg["train_dir"], **cfg["melspec_params"])
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
    model = AutomaticSpeechRecognition(
        n_residual_layers=3,
        n_rnn_layers=5,
        rnn_hidden_size=512,
        n_classes=n_classes,
        n_features=in_features
    )
    if cfg.get("state_dict_filepath") is not None:
        state_dict = torch.load(cfg.get("state_dict_filepath"), map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)

    loss_fn = nn.CTCLoss()
    optimizer = Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg["learning_rate"],
        steps_per_epoch=int(len(train_dataloader)),
        epochs=cfg["n_epochs"],
        anneal_strategy="linear"
    )

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
            scheduler=scheduler,
            criterion=loss_fn,
            device=device
        )

        info_valid = run_epoch(
            phase=VALID,
            epoch=epoch,
            model=model,
            dataloader=valid_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=loss_fn,
            device=device
        )

        if info_valid["loss"] <  best_metric:
            best_metric = info_valid["loss"]
            torch.save(model.state_dict(), best_model_filepath)
            mlflow.log_artifact(best_model_filepath)
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        torch.save(model.state_dict(), last_model_filepath)
        mlflow.log_artifact(last_model_filepath)

        if epochs_since_best > cfg["patience"]:
            break

    best_model = AutomaticSpeechRecognition(
        n_residual_layers=3,
        n_rnn_layers=5,
        rnn_hidden_size=512,
        n_classes=n_classes,
        n_features=in_features
    )
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_filepath")
    args = parser.parse_args()

    with open(args.config_filepath) as f:
        cfg = yaml.safe_load(f)

    with mlflow.start_run():
        mlflow.log_params(cfg)

        main(cfg)
