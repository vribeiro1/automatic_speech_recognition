import torch


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emissions):
        """
        Given a sequence emission over labels, get the best path.

        Args:
          emissions (Tensor): Logit tensors. Shape `[batch_size, seq_len, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        batch_indices = torch.argmax(emissions, dim=-1)
        batch_indices = torch.unique_consecutive(batch_indices, dim=-1)

        transcriptions = []
        for indices in batch_indices:
            indices = [i for i in indices if i != self.blank]
            transcription = "".join([self.labels[i.item()] for i in indices])
            transcriptions.append(transcription)

        return transcriptions
