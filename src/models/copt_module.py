from functools import partial
from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MaxMetric, MeanMetric

# from src.models.spaces import EVAL_FUNCTION_DICT, EVAL_FUNCTION_DICT_NOLABEL, LOSS_FUNCTION_DICT


class COPTModule(LightningModule):
    """LightningModule for Graph Convolutional Network (GCN).

    This module can be used for both graph-level and node-level tasks.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        task: str,
        metrics: Dict = None,
        labels: bool = False,
        compile: bool = False,
    ) -> None:
        """Initialize a `GCNLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param task: The task type, either "graph" or "node". Defaults to "graph".
        :param num_classes: The number of classes for classification. Defaults to 2.
        :param compile: Whether to compile the model. Defaults to False.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.task = task
        self.criterion = criterion

        self.metrics = metrics

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_metrics = {name: MeanMetric() for name in metrics}
        self.val_metrics = {name: MeanMetric() for name in metrics}
        self.test_metrics = {name: MeanMetric() for name in metrics}

        # for tracking best so far validation accuracy
        BestMetric = MinMetric if task in ['mds', 'mvc'] else MaxMetric
        self.val_best_metrics = {name: BestMetric() for name in metrics}

    def forward(self, batch):
        """Perform a forward pass through the model `self.net`.

        :param x: The input node features.
        :param edge_index: The edge indices.
        :param batch: The batch vector, which assigns each node to a specific example.
            Required for graph-level tasks. Defaults to None.
        :return: A tensor of logits.
        """
        return self.net(batch)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        for m in self.val_best_metrics:
            self.val_best_metrics[m].reset()

    def model_step(self, batch):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data from PyTorch Geometric.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        out = self.forward(batch)
        loss = self.criterion(out)
        return out, loss

    def training_step(self, batch, batch_idx):
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data from PyTorch Geometric.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        batch, loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch, batch_idx):
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data from PyTorch Geometric.
        :param batch_idx: The index of the current batch.
        """
        batch, loss = self.model_step(batch)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, metric_func in self.metrics.items():
            eval = metric_func(batch)
            self.val_metrics[name](eval)
            self.log(f"val/{name}", self.val_metrics[name], batch_size=batch.batch_size, on_epoch=True,
                     prog_bar=True, logger=True, metric_attribute=f"val_metrics.{name}")

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        for name, metric in self.val_metrics.items():
            current_val = metric.compute()  # get current metric
            self.val_best_metrics[name](current_val)  # update best so far metric
            # log metric as a value through `.compute()` method, instead of as a metric object
            # otherwise metric would be reset by lightning after each epoch
            self.log(f"val/{name}_best", self.val_best_metrics[name].compute(), sync_dist=True,
                     prog_bar=(name == "size"))

    def test_step(self, batch, batch_idx):
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data from PyTorch Geometric.
        :param batch_idx: The index of the current batch.
        """
        batch, loss = self.model_step(batch)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        for name, metric_func in self.metrics.items():
            eval = metric_func(batch)
            self.test_metrics[name](eval)
            self.log(f"test/{name}", self.test_metrics[name], batch_size=batch.batch_size, on_epoch=True,
                     prog_bar=True, logger=True, metric_attribute=f"test_metrics.{name}")

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = COPTModule(None, None, None)
