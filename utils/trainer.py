import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

class TrainerHarga(pl.LightningModule):
    
    def __init__(self) -> None:
        super().__init__()

    # Override dari abstract class LightningModule
    def configure_optimizers(self):
        optimizer = torch.optim.Adam()
        return optimizer

    # Override dari abstract class LightningModule
    def training_step(self, train_batch, batch_idx):
        return 0
    
    # Override dari abstract class LightningModule
    def test_step(self, test_batch, batch_idx):
        return 0
