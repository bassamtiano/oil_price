import torch
import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer) -> None:
        super().__init__()
        self.l1 = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layer = num_layer)
        
        self.linear = nn.Linear(hidden_size, 3)
        self.dropout = nn.Dropout()
        # self.l2 = nn.LSTM(
        #     input_size = input_size,
        #     hidden_size = hidden_size,
        #     num_layer = num_layer)
        
        self.activation = nn.Tanh()
        

    def forward(self, x_bbm, x_beras):
        output, hidden = self.l1(x_bbm)
        output = self.dropout(output)
        output = self.tanh(output)
        output = self.linear(output).squeeze()

        # output atau y-pred
        return output
        