"""DeepLOB (Zhang, Zohren & Roberts, IEEE Trans. Signal Processing, 2019).

Input: the last T = 100 order book snapshots, each 40 numbers in the FI-2010 order
(ask price, ask volume, bid price, bid volume) for levels 1..10, as a (batch, 1, T, 40)
"image". Output: logits for down / stationary / up.

  conv block 1  (1x2, stride 2) pairs each price with its volume       -> 20 columns
  conv block 2  (1x2, stride 2) pairs the ask side with the bid side   -> 10 columns
  conv block 3  (1x10) mixes all 10 levels into one "micro-structure" feature -> 1 column
  each block then has two (4x1) convolutions over time (no padding: T shrinks 100 -> 82)
  inception     1x1->3x1, 1x1->5x1, maxpool->1x1 in parallel: several time scales at once
  LSTM(192->64) over the 82 time steps; the last hidden state -> linear -> 3 logits

Same layers as the authors' PyTorch notebook, with one fix: the notebook applies softmax in
forward() and then nn.CrossEntropyLoss, which applies log-softmax again. We return logits.
"""
from __future__ import annotations

import torch
from torch import nn


def _conv(cin: int, cout: int, k: tuple[int, int], act: type[nn.Module], stride=(1, 1), padding=0):
    a = act(negative_slope=0.01) if act is nn.LeakyReLU else act()
    return [nn.Conv2d(cin, cout, k, stride=stride, padding=padding), a, nn.BatchNorm2d(cout)]



class DeepLOB(nn.Module):
    def __init__(self, n_classes: int = 3, conv_ch: int = 32, inc_ch: int = 64, lstm_hidden: int = 64,
                 n_features: int = 40):
        super().__init__()
        assert n_features == 40, "the 1x2, 1x2, 1x10 kernels assume 10 levels x 4 numbers"
        lrelu = nn.LeakyReLU
        self.conv1 = nn.Sequential(*_conv(1, conv_ch, (1, 2), lrelu, (1, 2)),
                                   *_conv(conv_ch, conv_ch, (4, 1), lrelu), *_conv(conv_ch, conv_ch, (4, 1), lrelu))
        self.conv2 = nn.Sequential(*_conv(conv_ch, conv_ch, (1, 2), nn.Tanh, (1, 2)),
                                   *_conv(conv_ch, conv_ch, (4, 1), nn.Tanh), *_conv(conv_ch, conv_ch, (4, 1), nn.Tanh))
        self.conv3 = nn.Sequential(*_conv(conv_ch, conv_ch, (1, 10), lrelu),
                                   *_conv(conv_ch, conv_ch, (4, 1), lrelu), *_conv(conv_ch, conv_ch, (4, 1), lrelu))
        self.inc1 = nn.Sequential(*_conv(conv_ch, inc_ch, (1, 1), lrelu),
                                  *_conv(inc_ch, inc_ch, (3, 1), lrelu, padding=(1, 0)))
        self.inc2 = nn.Sequential(*_conv(conv_ch, inc_ch, (1, 1), lrelu),
                                  *_conv(inc_ch, inc_ch, (5, 1), lrelu, padding=(2, 0)))
        self.inc3 = nn.Sequential(nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
                                  *_conv(conv_ch, inc_ch, (1, 1), lrelu))
        self.lstm = nn.LSTM(3 * inc_ch, lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:              # (batch, T, 40) -> (batch, 1, T, 40)
            x = x.unsqueeze(1)
        x = self.conv3(self.conv2(self.conv1(x)))                       # (B, 32, T-18, 1)
        x = torch.cat([self.inc1(x), self.inc2(x), self.inc3(x)], 1)    # (B, 192, T-18, 1)
        x = x.squeeze(-1).transpose(1, 2)                               # (B, T-18, 192)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])
