import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import pytorch_lightning as pl

from sklearn.preprocessing import MinMaxScaler

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

    # Menghitung jarak antara nilai hasil prediksi dengan nilai asli / kenyataan
    def calc_accuracy(self, pred, truth):
        # Menghitung rata rata
        dist = torch.mean(abs(pred) - abs(truth.squeeze()))
        # abs(pred) - abs(truth) nya di kurangi menggunakan operasi pengurangan matrix
        # torch.mean merata-ratakan hasil pengurangan operasi matrix

       # Karena trainer di training menggunakan gpu, agar dapat di tampilkan di log harus di pindahkan ke cpu
        dist = dist.detach().cpu()

        # konversi tensor matrix ke list
        dist = dist.tolist()
        return dist

    # Override dari abstract class LightningModule
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        h = tuple([each.data for each in self.hidden])
        
        output = self.model(x, h)
        
        dist = self.calc_accuracy(output, y)
        max_val = max(dist)

        scaler_acc = MinMaxScaler(feature_range = (0, 100))
        scaler_acc = scaler_acc.fit(dist)

        acc = scaler_acc.transform(dist)
        
        acc = np.mean(acc)

        self.log("Distance", dist, prog_bar = True)
        self.log("Accuracy", dist, prog_bar = True)

        loss = F.mse_loss(output, y.squeze())

        return loss
    
    # Override dari abstract class LightningModule
    def test_step(self, test_batch, batch_idx):
        return 0
