import pdb

import funcy
import numpy as np
import os
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as F

from glob import glob
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchaudio import transforms
from audio_preprocessing import dynamic_range_compression
from text_preprocessing import BLANK, SILENCE, UNKNOWN, text_processor


def collate_fn(batch):
    filepaths = [item[0] for item in batch]

    # Each melspec item has shape (channels, features, time)
    melspecs = [item[1].permute(2, 0, 1) for item in batch]
    melspec_lengths = torch.tensor(
        [melspec.shape[0] for melspec in melspecs],
        dtype=torch.int
    )
    padded_melspecs = pad_sequence(melspecs, batch_first=True, padding_value=-1)
    padded_melspecs = padded_melspecs.permute(0, 2, 3, 1)  # batch, channel, features, time

    transcriptions = [item[2] for item in batch]
    transcripts_lengths = torch.tensor(
        [transcription.shape[0] for transcription in transcriptions],
        dtype=torch.int
    )
    padded_transcriptions = pad_sequence(transcriptions, batch_first=True, padding_value=-1)

    return (
        filepaths,
        padded_melspecs,
        melspec_lengths,
        padded_transcriptions,
        transcripts_lengths
    )


class LibriSpeechDataset(Dataset):
    def __init__(
        self,
        datadir,
        blank=BLANK,
        silence=SILENCE,
        unknown=UNKNOWN,
        sample_rate=16000,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=80,
        f_min=0,
        f_max=None,
    ):
        super().__init__()

        self.datadir = datadir
        transcriptions_filepaths = glob(os.path.join(datadir, "*", "*", "*.txt"))

        data = []
        for filepath in transcriptions_filepaths:
            dirname = os.path.dirname(filepath)
            with open(filepath) as f:
                transcriptions = funcy.lmap(
                    lambda tup: (os.path.join(dirname, f"{tup[0]}.flac"), tup[1]),
                    map(lambda s: s.strip().split(" ", maxsplit=1), f.readlines())
                )

            data.extend(transcriptions)

        self.df = pd.DataFrame(data, columns=["filepath", "transcription"])
        self.vocabulary = self.make_vocabulary(
            self.df.transcription,
            blank,
            silence,
            unknown
        )
        self.sample_rate = sample_rate
        self.melspectrogram = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max
        )

    @staticmethod
    def make_vocabulary(transcriptions, blank, silence, unknown):
        all_tokens = set(funcy.flatten(map(list, transcriptions)))
        vocabulary = {blank: 0, silence: 1, unknown:2}
        vocabulary.update({token: i for i, token in enumerate(sorted(all_tokens), start=3)})
        return vocabulary

    @property
    def vocabulary_transposed(self):
        return {i: token for token, i in self.vocabulary.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        Return:
            filepath (str):
            melspec (torch.tensor): Tensor of shape (channels, features, time)
            tokens (torch.tensor): Tensor of shape (sequence_length,)
        """
        item = self.df.iloc[index]

        filepath = item["filepath"]
        transcription = item["transcription"]

        audio, sample_rate = torchaudio.load(filepath)
        audio = F.resample(audio, orig_freq=sample_rate, new_freq=self.sample_rate)
        audio = torch.concat([audio, audio], dim=0)
        melspec = dynamic_range_compression(self.melspectrogram(audio))

        tokens = torch.tensor(text_processor(transcription, self.vocabulary), dtype=torch.int)
        return_filepath = filepath.replace(self.datadir, "").strip("/")

        return return_filepath, melspec, tokens
