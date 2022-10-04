import torch
import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer) -> None:
        super().__init__()
        self.l1 = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layer = num_layer)
        self.l2 = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layer = num_layer)
        

    def forward(self, x1, x2):
        h1 = self.l1(x1)
        h2 = self.l2(x2)

        h = torch.cat((h1, h2), axis=1)
        