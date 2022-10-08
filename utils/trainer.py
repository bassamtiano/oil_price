import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from models.predictor import Predictor

class TrainerHarga(pl.LightningModule):
    
    def __init__(self) -> None:
        super().__init__()

        # Deklarasi model di trainer
        self.model = Predictor()

        h0 = torch.zeros((2, 1, 64))
        
        c0 = torch.zeros((2, 1, 64))

        # Memindahkan tensor matrix ke gpu
        # Hidden state
        h0 = h0.to(torch.device("cuda:0"))

        # Cell state (Output local)
        c0 = c0.to(torch.device("cuda:0"))

        self.hidden = (h0, c0)

    # Override dari abstract class LightningModule
    def configure_optimizers(self):
        optimizer = torch.optim.Adam()
        return optimizer

    def calc_accuracy(self, pred, truth):
        dist = torch.mean(abs(pred) - abs(truth.squeeze()))
        dist = dist.detach().cpu()
        dist = dist.tolist()
        return dist

    # Override dari abstract class LightningModule
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        h = tuple([each.data for each in self.hidden])
        
        output = self.model(x, h)
        
        dist = self.calc_accuracy(output, y)
        self.log("dist", dist, prog_bar = True)

        loss = F.mse_loss(output, y.squeze())

        return loss
    
    # Override dari abstract class LightningModule
    def test_step(self, test_batch, batch_idx):
        return 0
