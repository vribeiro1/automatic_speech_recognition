import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class AutomaticSpeechRecognition(nn.Module):
    def __init__(self, in_features, out_features, hidden_size=64):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=False,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * hidden_size),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_features)
        )

    def forward(self, x, lengths):
        packed_x = pack_padded_sequence(x, lengths, batch_first=False)
        packed_rnn_out, _ = self.rnn(packed_x)
        rnn_out, _ = pad_packed_sequence(packed_rnn_out, batch_first=False)

        outputs = self.classifier(rnn_out)

        return torch.log_softmax(outputs, dim=-1)
