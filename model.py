"""A small, fully convolutional model operating directly on the waveform."""

from torch import nn


class Transcription_CNN(nn.Module):
    """(batch, 1, N) waveform -> (batch, N, num_classes) probabilities.

    Use softmax for exclusive classes (monophonic events with silence).
    Use sigmoid for polyphony or all-zero silence targets.
    activation=None returns raw logits, useful for stable training losses.
    forward_logits is available independently of the selected activation.
    """

    stride = 1

    def __init__(self, num_classes=5, activation="softmax"):
        super().__init__()
        if (isinstance(num_classes, bool) or not isinstance(num_classes, int)
                or num_classes < 1):
            raise ValueError("num_classes must be a positive integer")
        if activation not in (None, "sigmoid", "softmax"):
            raise ValueError('activation must be "sigmoid", "softmax", or None')
        self.num_classes = num_classes
        self.activation = activation
        self.output_activation = {
            None: nn.Identity(), "sigmoid": nn.Sigmoid(), "softmax": nn.Softmax(dim=-1),
        }[activation]
        self.layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=self.stride, padding=5),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, num_classes, kernel_size=1),
        )

    def forward(self, waveform):
        return self.output_activation(self.forward_logits(waveform))

    def forward_logits(self, waveform):
        return self.layers(waveform).transpose(1, 2)
