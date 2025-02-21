import lightning.pytorch as pl
import torch
from torch.functional import F

from ..results.metrics import rms_calibration_error, sharpness
from ..utils.miscellaneous import enable_dropout, weights_init


class HNN(pl.LightningModule):
    """
    Pytorch Lightning frequentist models wrapper
    This class is used by frequentist models, MC_Dropout and Heteroscedastic NNs

    It implements various functions for Pytorch Lightning to manage train, test,
    validation, logging...
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        mc_samples: int,
        p_dropout: int,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.net.apply(weights_init)
        self.net.dropout = p_dropout
        self.net.mc_samples = int(mc_samples)
        self.validation_step_outputs = []

    def forward(self, grp_descrp, logic_reduce):
        return self.net.forward(grp_descrp, logic_reduce)

    def get_device(self):
        return next(self.net.parameters()).device

    def to_device(self, device: torch.device):
        self.net.to(device)

    def step(self, batch, phase):
        grp_descrp = batch[10]
        grp_energy = batch[11]
        logic_reduce = batch[12]
        # grp_N_atom = batch[14]
        output = self.forward(grp_descrp, logic_reduce)
        loc, scale = output.chunk(2, dim=-1) 

        if phase == "predict":
            return loc, scale
        loss = F.gaussian_nll_loss(loc, grp_energy, torch.square(scale))
        self.log(f"nll/{phase}", loss, on_step=False, on_epoch=True, batch_size=len(batch[11]))
        return loss, loc, scale

    def training_step(self, batch, batch_idx):
        loss, loc, scale = self.step(batch, "train")
        mse = F.mse_loss(loc, batch[11])
        rmsce = rms_calibration_error(loc, scale, batch[11])
        sharp = sharpness(scale)
        self.log("mse/train", mse, on_step=False, on_epoch=True, batch_size=len(batch[11]))
        self.log("rmsce/train", rmsce, on_step=False, on_epoch=True, batch_size=len(batch[11]))
        self.log("sharp/train", sharp, on_step=False, on_epoch=True, batch_size=len(batch[11]))
        return loss

    def mc_sampling(self, batch, mc_samples: int, phase: str, agg: bool = True):
        losses = []
        locs = []
        scales = []
        for _ in range(int(mc_samples)):
            if phase == "predict":
                loc, scale = self.step(batch, phase)
            else:
                loss, loc, scale = self.step(batch, phase)
                losses.append(loss)
            locs.append(loc)
            scales.append(scale)
        locs = torch.stack(locs)
        scales = torch.stack(scales)
        if phase == "predict":
            return locs, scales
        loss = torch.stack(losses).mean(0)
        if agg:
            scale = scales.pow(2).mean(0).add(locs.var(0)).sqrt()
            loc = locs.mean(0)
            return loss, loc, scale
        return loss, locs, scales

    def validation_step(self, batch, batch_idx):
        phase = "val"
        if self.net.dropout > 0:
            enable_dropout(self.net)
            loss, loc, scale = self.mc_sampling(
                batch, self.hparams.mc_samples, phase=phase
            )
        else:
            loss, loc, scale = self.step(batch, phase)
        self.validation_step_outputs.append({"loss": loss, "label": batch[11], "pred": loc, "std": scale})
        return {"loss": loss, "label": batch[11], "pred": loc, "std": scale}

    def on_validation_epoch_end(self) -> None:
        for i, output in enumerate(self.validation_step_outputs):
            if i == 0:
                preds = output["pred"].detach()
                labels = output["label"].detach()
                stds = output["std"].detach()
            else:
                preds = torch.cat([preds, output["pred"].detach()])
                labels = torch.cat([labels, output["label"].detach()])
                stds = torch.cat([stds, output["std"].detach()])

        mse = F.mse_loss(preds, labels)
        rmsce = rms_calibration_error(preds, stds, labels)
        sharp = sharpness(stds)
        self.log("mse/val", mse, batch_size=len(labels))
        self.log("rmsce/val", rmsce, batch_size=len(labels))
        self.log("sharp/val", sharp, batch_size=len(labels))

    def test_step(self, batch, batch_idx):
        y = batch[11]
        phase = "test"
        if self.net.dropout > 0:
            enable_dropout(self.net)
            loss, locs, scales = self.mc_sampling(
                batch, self.hparams.mc_samples, phase=phase, agg=False
            )
            ep_var = locs.var(0)
            al_var = (scales**2).mean(0)
            scale = al_var.add(ep_var).sqrt()
            loc = locs.mean(axis=0)
        else:
            loss, loc, scale = self.step(batch, phase)

        self.log("nll/test", loss, batch_size=len(y))
        self.log("mse/test", F.mse_loss(loc, y), batch_size=len(y))
        self.log("rmsce/test", rms_calibration_error(loc, scale, y), batch_size=len(y))
        self.log("sharp/test", sharpness(scale), batch_size=len(y))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pred = dict()
        pred["labels"] = batch[11].cpu().numpy()
        phase = "predict"
        if self.net.dropout > 0:
            enable_dropout(self.net)
            locs, scales = self.mc_sampling(
                batch, self.hparams.mc_samples, phase=phase, agg=False
            )
            ep_var = locs.var(0)
            al_var = (scales**2).mean(0)
            scale = al_var.add(ep_var).sqrt()
            loc = locs.mean(axis=0)
            pred["ep_vars"] = ep_var.cpu().numpy()
            pred["al_vars"] = al_var.cpu().numpy()
        else:
            loc, scale = self.step(batch, phase)
        pred["preds"] = loc.cpu().numpy().astype(float)
        pred["stds"] = scale.cpu().numpy().astype(float)
        pred["n_atoms"] = batch[14].cpu().numpy().astype(float)
        return pred

    def configure_optimizers(self):
        return self.hparams.optimizer(params=self.parameters())

