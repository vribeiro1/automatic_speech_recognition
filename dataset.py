import pdb
import funcy
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
from text_preprocessing import BLANK, SILENCE, text_processor


def collate_fn(batch):
    filepaths = [item[0] for item in batch]

    melspecs = [item[1].permute(2, 0, 1) for item in batch]
    lengths = torch.tensor([melspec.shape[0] for melspec in melspecs], dtype=torch.int)
    melspecs_lenghts_sorted, melspecs_sort_indices = lengths.sort(descending=True)
    padded_melspecs = pad_sequence(melspecs, batch_first=False)
    padded_melspecs = padded_melspecs[:, melspecs_sort_indices, :]  # sequence_length, batch, channels, features
    padded_melspecs = padded_melspecs.permute(1, 2, 3, 0)  # batch, channel, features, sequence_length

    transcriptions = [item[2] for item in batch]
    lengths = torch.tensor([transcription.shape[0] for transcription in transcriptions], dtype=torch.int)
    transcripts_lengths_sorted, transcripts_sort_indices = lengths.sort(descending=True)
    padded_transcriptions = pad_sequence(transcriptions, batch_first=False)
    padded_transcriptions = padded_transcriptions[:, transcripts_sort_indices]
    padded_transcriptions = padded_transcriptions.permute(1, 0)

    return (
        filepaths,
        padded_melspecs,
        melspecs_lenghts_sorted,
        padded_transcriptions,
        transcripts_lengths_sorted
    )


class LibriSpeechDataset(Dataset):
    def __init__(self, datadir, blank=BLANK, silence=SILENCE, sample_rate=16000, n_fft=1024, win_length=1024, hop_length=256, n_mels=80, f_min=0, f_max=None):
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
        self.vocabulary = self.make_vocabulary(self.df.transcription, blank, silence)
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
    def make_vocabulary(transcriptions, blank, silence):
        all_tokens = set(funcy.flatten(map(list, transcriptions)))
        vocabulary = {blank: 0}
        vocabulary.update({token: i for i, token in enumerate(sorted(all_tokens), start=1)})
        vocabulary[silence] = len(vocabulary)
        return vocabulary

    @property
    def vocabulary_transposed(self):
        return {i: token for token, i in self.vocabulary.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]

        filepath = item["filepath"]
        transcription = item["transcription"]

        audio, sample_rate = torchaudio.load(filepath)
        audio = F.resample(audio, orig_freq=sample_rate, new_freq=self.sample_rate)
        melspec = dynamic_range_compression(self.melspectrogram(audio))

        tokens = torch.tensor(text_processor(transcription, self.vocabulary), dtype=torch.int)

        return_filepath = filepath.replace(self.datadir, "").strip("/")

        return return_filepath, melspec, tokens
