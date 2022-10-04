import torch
import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x1, x2):
        