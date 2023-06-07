import pdb

import argparse
import logging
import mlflow
import numpy as np
import os
import tempfile
import torch
import torch.nn as nn
import yaml
import shutil

from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader
from torchaudio.models.decoder import ctc_decoder
from torchmetrics.functional import word_error_rate
from tqdm import tqdm

from dataset import LibriSpeechDataset, collate_fn
from helpers import set_seeds
from model import DeepSpeech2
from text_preprocessing import BLANK, SILENCE, UNKNOWN

TRAIN = "train"
VALID = "validation"
TEST = "test"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TMPFILES = os.path.join(BASE_DIR, "tmp")
TMP_DIR = tempfile.mkdtemp(dir=TMPFILES)
RESULTS_DIR = os.path.join(TMP_DIR, "results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


def run_epoch(
    phase,
    epoch,
    model,
    dataloader,
    optimizer,
    criterion,
    logits_large_margins=0.0,
    fn_metrics=None,
    scheduler=None,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = phase == TRAIN
    if fn_metrics is None:
        fn_metrics = {}

    if training:
        model.train()
    else:
        model.eval()

    losses = []
    metrics_values = {
        metric_name: [] for metric_name, fn_metric in fn_metrics.items()
    }
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - {phase}")
    for i, (_, melspecs, len_melspecs, transcripts, len_transcripts) in enumerate(progress_bar, start=epoch * int(len(dataloader))):
        melspecs = melspecs.to(device)
        transcripts = transcripts.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            outputs = model(melspecs, len_melspecs)
            if training and logits_large_margins > 0.0:
                outputs = model.get_noise_logits(outputs, logits_large_margins)
            outputs = model.get_normalized_outputs(outputs, use_log_prob=True)
            loss = criterion(
                outputs.permute(1, 0, 2),
                transcripts,
                len_melspecs,
                len_transcripts
            )

            if training:
                loss.backward()
                optimizer.step()

                if scheduler:
                    scheduler.step()

            for metric_name, fn_metric in fn_metrics.items():
                metric_val = fn_metric(outputs, transcripts, len_melspecs, len_transcripts)
                metrics_values[metric_name].append(metric_val.item())
            losses.append(loss.item())

        mean_loss = np.mean(losses)
        postfix = {"loss": mean_loss}
        postfix.update({
            metric_name: np.mean(metric_val)
            for metric_name, metric_val in metrics_values.items()
        })
        progress_bar.set_postfix(**postfix)

    mean_loss = np.mean(losses)
    info = {
        "loss": mean_loss
    }
    info.update({
        metric_name: np.mean(metric_val)
        for metric_name, metric_val in metrics_values.items()
    })

    mlflow.log_metrics({
        f"{phase}_{metric}": mean_metric
        for metric, mean_metric in info.items()
    })

    return info


def run_test(phase, epoch, model, dataloader, criterion, fn_metrics, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    losses = []
    metrics_values = {
        metric_name: [] for metric_name, fn_metric in fn_metrics.items()
    }
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - {phase}")
    for _, melspecs, len_melspecs, transcripts, len_transcripts in progress_bar:
        melspecs = melspecs.to(device)
        transcripts = transcripts.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(melspecs, len_melspecs)
            loss = criterion(
                outputs.permute(1, 0, 2),
                transcripts,
                len_melspecs,
                len_transcripts
            )
            for metric_name, fn_metric in fn_metrics.items():
                metric_val = fn_metric(outputs, transcripts, len_melspecs, len_transcripts)
                metrics_values[metric_name].append(metric_val.item())

            losses.append(loss.item())

        mean_loss = np.mean(losses)
        postfix = {"loss": mean_loss}
        postfix.update({
            metric_name: np.mean(metric_val)
            for metric_name, metric_val in metrics_values.items()
        })
        progress_bar.set_postfix(**postfix)
        progress_bar.set_postfix(loss=np.mean(losses))

    mean_loss = np.mean(losses)
    info = {
        "loss": mean_loss,
    }
    info.update({
        metric_name: np.mean(metric_val)
        for metric_name, metric_val in metrics_values.items()
    })

    return info


class EditDistance:
    def __init__(self, decoder):
        self.decoder = decoder
        self.reduction = torch.mean

    def __call__(self, emissions, targets, input_lengths, target_lengths):
        emissions = emissions  # (B, T, C)
        emissions = emissions.detach().cpu()

        targets = targets.detach().cpu()  # (B, T)
        target_sequences = []
        for target, length in zip(targets, target_lengths):
            target_no_pad = target[:length]
            tokens = [str(token.item()) for token in target_no_pad]
            target_sequences.append(" ".join(tokens))

        results = self.decoder(emissions, input_lengths)
        pred_sequences = []
        for result in results:
            best_hyp = result[0]
            tokens = [str(token.item()) for token in best_hyp.tokens]
            pred_sequences.append(" ".join(tokens))

        edit_dist = word_error_rate(pred_sequences, target_sequences)
        return edit_dist


def main(
    train_dir,
    valid_dir,
    test_dir,
    num_epochs,
    patience,
    batch_size,
    learning_rate,
    weight_decay,
    melspec_params,
    model_params,
    num_workers=0,
    logits_large_margins=0,
    state_dict_filepath=None,
    checkpoint_filepath=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on '{device.type}'")

    best_model_filepath = os.path.join(TMP_DIR, "best_model.pt")
    last_model_filepath = os.path.join(TMP_DIR, "last_model.pt")
    save_checkpoint_filepath = os.path.join(TMP_DIR, "checkpoint.pt")

    train_dataset = LibriSpeechDataset(
        train_dir,
        **melspec_params,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=set_seeds,
    )

    valid_dataset = LibriSpeechDataset(
        valid_dir,
        **melspec_params,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=set_seeds,
    )

    test_dataset = LibriSpeechDataset(
        test_dir,
        **melspec_params,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=set_seeds,
    )

    num_classes = len(train_dataset.vocabulary)
    model = DeepSpeech2(
        num_classes=num_classes,
        **model_params,
    )
    if state_dict_filepath is not None:
        state_dict = torch.load(state_dict_filepath, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)

    loss_fn = nn.CTCLoss(
        blank=0,
        zero_infinity=True,
    )
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = CyclicLR(
        optimizer,
        base_lr=learning_rate / 25,
        max_lr=learning_rate,
        cycle_momentum=False
    )

    decoder = ctc_decoder(
        lexicon=None,
        tokens=list(train_dataset.vocabulary.keys()),
        blank_token=BLANK,
        sil_token=SILENCE,
        unk_word=UNKNOWN,
    )
    metrics = {
        "edit_distance": EditDistance(decoder)
    }

    logits_large_margins = logits_large_margins
    epochs = range(1, num_epochs + 1)
    best_metric = torch.inf
    epochs_since_best = 0

    if checkpoint_filepath is not None:
        checkpoint = torch.load(checkpoint_filepath)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # scheduler.load_state_dict(checkpoint["scheduler"])
        epochs = range(checkpoint["epoch"], num_epochs + 1)
        best_metric = checkpoint["best_metric"]
        epochs_since_best = checkpoint["epochs_since_best"]

        print(f"""
Loaded checkpoint -- Launching training from epoch {epoch} with best metric
so far {best_metric} seen {epochs_since_best} epochs ago.
""")

    for epoch in epochs:
        info_train = run_epoch(
            phase=TRAIN,
            epoch=epoch,
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            logits_large_margins=logits_large_margins,
            criterion=loss_fn,
            device=device
        )

        valid_metrics = metrics if info_train["loss"] < 1. else None
        info_valid = run_epoch(
            phase=VALID,
            epoch=epoch,
            model=model,
            dataloader=valid_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=loss_fn,
            fn_metrics=valid_metrics,
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

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            # "scheduler": scheduler.state_dict(),
            "best_metric": best_metric,
            "epochs_since_best": epochs_since_best
        }
        torch.save(checkpoint, save_checkpoint_filepath)
        mlflow.log_artifact(save_checkpoint_filepath)

        print(f"""
Finished training epoch {epoch}
Best metric: {'%0.4f' % best_metric}, Epochs since best: {epochs_since_best}
""")

        if epochs_since_best > patience:
            break

    best_model = DeepSpeech2(
        num_classes=num_classes,
        **model_params,
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
        fn_metrics=metrics,
        device=device,
    )
    print(info_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_filepath")
    parser.add_argument("--mlflow", dest="mlflow_tracking_uri", default=None)
    parser.add_argument("--experiment", dest="experiment_name", default="phoneme_recognition")
    parser.add_argument("--run_id", dest="run_id", default=None)
    parser.add_argument("--run_name", dest="run_name", default=None)
    parser.add_argument("--checkpoint", dest="checkpoint_filepath", default=None)
    args = parser.parse_args()

    if args.mlflow_tracking_uri is not None:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    with open(args.config_filepath) as f:
        cfg = yaml.safe_load(f)

    experiment = mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(
        run_id=args.run_id,
        experiment_id=experiment.experiment_id,
        run_name=args.run_name
    ) as run:
        print(f"Experiment ID: {experiment.experiment_id}\nRun ID: {run.info.run_id}")
        try:
            mlflow.log_params(cfg)
            mlflow.log_artifact(args.config_filepath)
        except shutil.SameFileError:
            logging.info("Skipping logging config file since it already exists.")
        try:
            main(
                **cfg,
                checkpoint_filepath=args.checkpoint_filepath,
            )
        finally:
            shutil.rmtree(TMP_DIR)
