import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class CNNLayerNorm(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.layer_norm = nn.LayerNorm(n_features)

    def forward(self, x):
        """
        x: torch.tensor of shape (batch_size, channels, features, sequence_len)
        """
        out = x.transpose(2, 3)  #


class ResidualCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_features):
        super().__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel // 2)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.layer_norm1 = nn.LayerNorm(n_features)
        self.layer_norm2 = nn.LayerNorm(n_features)

    def forward(self, x):
        """
        x: torch.tensor of shape (batch_size, channels, features, sequence_len)
        """
        out = self.layer_norm1(x.transpose(2, 3)).transpose(3, 2)
        out = F.gelu(out)
        out = self.dropout1(out)
        out = self.cnn1(out)

        out = self.layer_norm2(out.transpose(2, 3)).transpose(3, 2)
        out = F.gelu(out)
        out = self.dropout2(out)
        out = self.cnn2(out)

        out += x

        return out


class RecurrentBlock(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=embed_size, hidden_size=hidden_size, num_layers=1, bidirectional=True, batch_first=False
        )
        self.layer_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        out = self.layer_norm(x)
        out = F.gelu(out)
        packed_out = pack_padded_sequence(x, lengths, batch_first=False)
        packed_rnn_out, _ = self.rnn(packed_out)
        rnn_out, _ = pad_packed_sequence(packed_rnn_out, batch_first=False)
        out = self.dropout(rnn_out)

        return out


class AutomaticSpeechRecognition(nn.Module):
    def __init__(self, n_residual_layers, n_rnn_layers, rnn_hidden_size, n_classes, n_features, dropout=0.1):
        super().__init__()

        # n_features = n_features // 2
        in_channels = 1
        out_channels = 32
        kernel = 3
        padding = kernel // 2
        self.cnn = nn.Conv2d(in_channels, out_channels, kernel, stride=1, padding=padding)

        self.residual_layers = nn.ModuleList([
            ResidualCNN(out_channels, out_channels, kernel=kernel, stride=1, dropout=dropout, n_features=n_features)
            for _ in range(n_residual_layers)
        ])

        self.linear = nn.Linear(n_features * out_channels, rnn_hidden_size)
        self.recurrent_layers = nn.ModuleList([
            RecurrentBlock(rnn_hidden_size if i == 0 else 2 * rnn_hidden_size, rnn_hidden_size, dropout=dropout)
            for i in range(n_rnn_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(2 * rnn_hidden_size, rnn_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden_size, n_classes)
        )

    def forward(self, x, lengths):
        out = self.cnn(x)

        for residual_layer in self.residual_layers:
            out = residual_layer(out)

        batch, channels, features, sequence_len = out.shape
        out = out.view(batch, channels * features, sequence_len)
        out = out.permute(2, 0, 1)  # sequence_len, batch, features
        out = self.linear(out)

        for recurrent_layer in self.recurrent_layers:
            out = recurrent_layer(out, lengths)
        out = self.classifier(out)
        out = torch.log_softmax(out, dim=-1)

        return out
