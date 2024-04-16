import pytorch_lightning as pl
import torch
from torch.functional import F

from ..utils.miscellaneous import enable_dropout, weights_init

class NN(pl.LightningModule):
    """
    Class used by BNNs to pretrain their weights. This class is instantiated,
    trained for X epochs and then it stores its weights in the log directory.
    VIBnnWrapper then loads the weights and starts the Bayesian training
    """

    def __init__(self, net: torch.nn.Module, optimizer: torch.optim.Optimizer):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.net.apply(weights_init)

    def forward(self, x):
        return self.net(x)

    def step(self, batch):
        x = batch[10], batch[12]
        y = batch[11]
        output = self.net(x)
        y_hat = output[:, 0]
        return F.mse_loss(y_hat, y.squeeze())

    def training_step(self, batch, batch_idx):
        return self.step(batch)

    def validation_step(self, batch, batch_idx):
        mse = self.step(batch)
        self.log("mse/val", mse, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return self.hparams.optimizer(params=self.parameters())