from functools import partial
from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MaxMetric, MeanMetric
from typing import Dict, List, Callable, Union, Optional
import warnings
import torch.nn as nn
from src.models.discretizer import IMLESampler
from torch_geometric.data import Batch


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
        discretize: bool = False
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
        BestMetric = MinMetric if task in ['mds', 'mvc', 'color', 'hcp'] else MaxMetric
        self.val_best_metrics = {name: BestMetric() for name in metrics}

        #discretizer
        self.discretizer=None
        if discretize:
            self.discretizer = IMLESampler(device=self.device)

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
        pre_out = self.forward(batch)

        if self.discretizer is not None:
            discrete_data, _ = self.discretizer(pre_out.x, 1)
            new_graphs = Batch.from_data_list(Batch.to_data_list(batch))
            discrete_data = discrete_data.flatten(0, 1) if self.training else discrete_data.squeeze().detach()
            new_graphs.x = discrete_data
            out = new_graphs
        else:
            out = pre_out
        
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

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data from PyTorch Geometric.
        :param batch_idx: The index of the current batch.
        :param dataloader_idx: Index of the current dataloader (for multiple test dataloaders).
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


class COPTTransferModule(COPTModule):
    """LightningModule for transfer learning with pretrained GNN/HybridGNN models.

    This module loads a pretrained GNN/HybridGNN from a checkpoint and freezes
    all parameters except the output head (post_mp/GNNHead) for transfer learning.

    This module can be used for both graph-level and node-level tasks.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        pretrain_path: str,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        task: str,
        metrics: Dict = None,
        labels: bool = False,
        compile: bool = False,
        freeze: bool = False,
        invert_head: bool = False,
        reset_head: bool = True,
        reset_encoder: bool = False,
    ) -> None:
        """Initialize a `COPTTransferModule`.

        :param net: The model to train (GNN or HybridGNN). Will be loaded with pretrained weights.
        :param pretrain_path: Path to the pretrained checkpoint file.
        :param criterion: The loss function.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param task: The task type, either "graph" or "node". Defaults to "graph".
        :param metrics: Dictionary of metrics to compute.
        :param labels: Whether labels are provided. Defaults to False.
        :param compile: Whether to compile the model. Defaults to False.
        :param freeze_backbone: Whether to freeze the backbone (everything except post_mp).
            Defaults to True.
        :param reset_head: Whether to reset the head (post_mp) weights. If True, head will be
            randomly initialized. If False, pretrained head weights will be loaded. Defaults to True.
        """
        self._load_pretrained_weights(net, pretrain_path, invert_head=invert_head, reset_head=reset_head, reset_encoder=reset_encoder)
        if freeze == 'backbone':
            self._freeze_backbone(net)
        elif freeze == 'gnn_stack':
            self._freeze_gnn_stack(net)
        elif freeze == 'all':
            for name, param in net.named_parameters():
                param.requires_grad = False
        elif freeze:
            raise ValueError('freeze must be either False, "backbone" or "all"')
        
        super().__init__(
            net=net,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            task=task,
            metrics=metrics,
            labels=labels,
            compile=compile,
        )

    def _load_pretrained_weights(self, net: torch.nn.Module, pretrain_path: str, invert_head: bool = False, reset_head: bool = True, reset_encoder: bool = False) -> None:
        """Load pretrained weights from a checkpoint into the network.
        
        :param net: The network to load weights into.
        :param pretrain_path: Path to the checkpoint file.
        :param reset_head: If True, exclude post_mp weights to allow head reset. If False, load all weights.
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(pretrain_path, map_location=device, weights_only=False)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        net_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('net.'):
                # Remove 'net.' prefix
                new_key = key[4:]  # Remove 'net.' prefix
                net_state_dict[new_key] = value
        
        # If no 'net.' prefix found, try direct matching
        if not net_state_dict:
            # Try to match keys directly
            model_keys = set(net.state_dict().keys())
            for key, value in state_dict.items():
                if key in model_keys:
                    net_state_dict[key] = value
        
        if net_state_dict:
            model_dict = net.state_dict()
            # Filter out post_mp keys if resetting head
            # Also skip gt_stack keys (new component, will be randomly initialized)
            if invert_head:
                filtered_pretrained_dict = {}
                assert not (reset_head or reset_encoder)
                for k, v in net_state_dict.items():
                    if k.startswith('post_mp'):
                        filtered_pretrained_dict[k] = -v
                    else:
                        filtered_pretrained_dict[k] = v
            elif reset_head:
                if reset_encoder:
                    filtered_pretrained_dict = {
                        k: v for k, v in net_state_dict.items()
                        if k in model_dict and not k.startswith('encoder') and not k.startswith('pre_mp')
                           and not k.startswith('post_mp') and not k.startswith('gt_stack')
                    }
                else:
                    filtered_pretrained_dict = {
                        k: v for k, v in net_state_dict.items()
                        if k in model_dict and not k.startswith('post_mp') and not k.startswith('gt_stack')
                    }
            else:
                # Load all matching keys including post_mp, but skip gt_stack
                filtered_pretrained_dict = {
                    k: v for k, v in net_state_dict.items()
                    if k in model_dict and not k.startswith('gt_stack')
                }
            model_dict.update(filtered_pretrained_dict)
            net.load_state_dict(model_dict, strict=False)
        else:
            raise ValueError(f"Could not extract network weights from checkpoint at {pretrain_path}")

    def _freeze_backbone(self, net: torch.nn.Module) -> None:
        """Freeze backbone (encoder, pre_mp, mp), keep gt_stack and post_mp trainable.
        
        :param net: The network to freeze.
        """
        for name, param in net.named_parameters():
            # Keep gt_stack and post_mp trainable, freeze everything else (backbone)
            if not (name.startswith('gt_stack') or name.startswith('post_mp')):
                param.requires_grad = False

    def _freeze_gnn_stack(self, net: torch.nn.Module) -> None:
        """Freeze backbone (encoder, pre_mp, mp), keep gt_stack and post_mp trainable.

        :param net: The network to freeze.
        """
        for name, param in net.named_parameters():
            # Freeze only the GNN layers
            if not ( name.startswith('encoder') or name.startswith('pre_mp')
                     or name.startswith('gt_stack') or name.startswith('post_mp')):
                param.requires_grad = False


#if __name__ == "__main__":
#    _ = COPTModule(None, None, None)



class MultiCOPTModule(LightningModule):
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
        criterion: Dict[str, Callable],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        task: List[str] = None, #now a list of tasks
        metrics: Optional[Dict[str, Dict[str, Callable]]] = None,
        labels: bool = False,
        compile: bool = False,
        weights: Optional[Dict[str,float]] = None,
        strategy: str = 'sum',
        discretize: bool = False
    ) -> None:
        """Initialize a `GCNLitModule`.

        :param net: The model to train.
        :param criterion: Dictionary of loss functions
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param tasks: List of tasks
        :param metrics: Dictionary of metrics
        :param num_classes: The number of classes for classification. Defaults to 2.
        :param compile: Whether to compile the model. Defaults to False.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.tasks = self.net.tasks
        self.criterion = {task : criterion[task] for task in self.tasks} 
        self.metrics = {task : metrics[task] for task in self.tasks}  
        
        #discretizer
        self.discretizer=None
        if discretize:
            self.discretizer = IMLESampler(device=self.device)

        # MTL strategy
        if weights is None:
            self.weights = {task : 1.0 for task in self.tasks}
        else:
            self.weights = {task : weights[task] for task in self.tasks}

        if strategy not in ['sum', 'alternate', 'pcgrad']:
            raise ValueError(f"Invalid strategy '{strategy}'. Must be one of {['sum', 'alternate', 'pcgrad']}")    
        self.strategy = strategy
        self.current_task_idx = 0
        self.task_list = list(self.tasks)
        if strategy == 'pcgrad':
            self.automatic_optimization = False

        # for averaging loss across batches
        device = self.device

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_losses = nn.ModuleDict({task : MeanMetric() for task in self.tasks})
        self.val_losses = nn.ModuleDict({task : MeanMetric() for task in self.tasks})
        self.test_losses = nn.ModuleDict({task : MeanMetric() for task in self.tasks})

        self.train_metrics = {
            task: {name: MeanMetric() for name in task_metrics}
            for task, task_metrics in self.metrics.items()
        }

        self.val_metrics = {
            task: {name: MeanMetric() for name in task_metrics}
            for task, task_metrics in self.metrics.items()
        }

        self.test_metrics = {
            task: {name: MeanMetric() for name in task_metrics}
            for task, task_metrics in self.metrics.items()
        }

        # for tracking best so far validation accuracy
        MIN_TASKS = ['mds','color','mvc','hcp']
        self.val_best_metrics = {
            task: {name: (MinMetric() if task in MIN_TASKS else MaxMetric()) for name in task_metrics}
            for task, task_metrics in self.metrics.items()
        }

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
        for task_metrics in self.val_best_metrics.values():
            for metric in task_metrics.values():
                metric.reset()

    def _pcgrad_step(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply PCGrad to resolve conflicting gradients.
        
        :param losses: Dictionary of task losses
        :return: Combined loss after gradient projection
        """
        # Store original parameters
        self.net.zero_grad()
        
        # Compute gradients for each task
        task_gradients = {}
        for task, loss in losses.items():
            # Compute gradients for this task
            self.net.zero_grad()
            loss.backward(retain_graph=True)
            
            # Store gradients
            task_gradients[task] = []
            for param in self.net.parameters():
                if param.grad is not None:
                    task_gradients[task].append(param.grad.clone())
                else:
                    task_gradients[task].append(torch.zeros_like(param))
        
        # Apply PCGrad: project conflicting gradients
        self.net.zero_grad()
        projected_grads = {task: [g.clone() for g in grads] 
                          for task, grads in task_gradients.items()}
        
        # For each pair of tasks, project if conflicting
        task_names = list(self.tasks)
        for i, task_i in enumerate(task_names):
            for j, task_j in enumerate(task_names):
                if i == j:
                    continue
                
                # Compute dot product between gradient vectors
                dot_product = sum(
                    (g_i * g_j).sum()
                    for g_i, g_j in zip(task_gradients[task_i], task_gradients[task_j])
                )
                
                # If conflicting (negative dot product), project
                if dot_product < 0:
                    # Project task_i's gradient away from task_j's gradient
                    g_j_norm_sq = sum((g ** 2).sum() for g in task_gradients[task_j])
                    if g_j_norm_sq > 0:
                        proj_coef = dot_product / g_j_norm_sq
                        projected_grads[task_i] = [
                            g_i - proj_coef * g_j
                            for g_i, g_j in zip(projected_grads[task_i], task_gradients[task_j])
                        ]
        
        # Average the projected gradients with weights
        final_grads = []
        for param_idx, param in enumerate(self.net.parameters()):
            weighted_grad = sum(
                self.weights[task] * projected_grads[task][param_idx]
                for task in self.tasks
            )
            final_grads.append(weighted_grad)
        
        # Set the final gradients
        for param, grad in zip(self.net.parameters(), final_grads):
            param.grad = grad
        
        # Return weighted average of losses for logging
        return sum(self.weights[task] * losses[task] for task in self.tasks)

    def model_step(self, batch, training=True):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data from PyTorch Geometric.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """

        pre_out = self.forward(batch)

        if self.discretizer is not None:
            out = {}
            for task in self.tasks:
                discrete_data, _ = self.discretizer(pre_out[task].x, 1)
                new_graphs = Batch.from_data_list(Batch.to_data_list(batch))
                discrete_data = discrete_data.flatten(0, 1) if self.training else discrete_data.squeeze().detach()
                new_graphs.x = discrete_data
                out[task] = new_graphs
        else:
            out = pre_out
            
        losses = {
            task: self.criterion[task](out[task]) 
            for task in self.tasks
        }

        if training:
            if self.strategy == 'sum':
                loss = sum(self.weights[task] * losses[task] for task in self.tasks)
            elif self.strategy == 'alternate':
                current_task = self.task_list[self.current_task_idx]
                loss = losses[current_task]
            elif self.strategy == 'pcgrad':
                loss = sum(self.weights[task] * losses[task] for task in self.tasks)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
        else:
            # During validation/testing, always sum all tasks
            loss = sum(self.weights[task] * losses[task] for task in self.tasks)

        return out, losses, loss
    
    '''
    def training_step(self, batch, batch_idx):
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data from PyTorch Geometric.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        batch, losses, loss = self.model_step(batch)
        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        for task,task_loss in losses.items():
            self.train_losses[task](task_loss)
            self.log(f"train/{task}/loss", self.train_losses[task], on_step=False, on_epoch=True, prog_bar=True,
                     metric_attribute=f"train_losses.{task}")

        # return loss or backpropagation will fail
        return loss
    '''

    def training_step(self, batch, batch_idx):
        """Perform a single training step.
        
        :param batch: A batch of data
        :param batch_idx: The index of the current batch
        :return: Loss tensor for backpropagation
        """
        if self.strategy == 'pcgrad':
            opt = self.optimizers()
            opt.zero_grad()
            
            out = self.forward(batch)
            losses = {task: self.criterion[task](out[task]) for task in self.tasks}
            
            loss = self._pcgrad_step(losses)
            opt.step()
            
            # Step scheduler if it exists
            sch = self.lr_schedulers()
            if sch is not None:
                sch.step()
            
            # Update metrics
            self.train_loss(loss.detach())
            for task, task_loss in losses.items():
                self.train_losses[task](task_loss.detach())
            
            return loss.detach()
        else:
            # Standard forward pass
            batch_out, losses, loss = self.model_step(batch,training=True)
            
            # Update metrics
            self.train_loss(loss)
            for task, task_loss in losses.items():
                self.train_losses[task](task_loss)
            
            # For alternating strategy, update task index
            if self.strategy == 'alternate':
                self.current_task_idx = (self.current_task_idx + 1) % len(self.task_list)
            
            return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Called after training_step. Log metrics here."""
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        for task in self.tasks:
            self.log(
                f"train/{task}/loss",
                self.train_losses[task],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                metric_attribute=f"train_losses.{task}"
            )
    
    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch, batch_idx):
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data from PyTorch Geometric.
        :param batch_idx: The index of the current batch.
        """
        batch, losses, loss = self.model_step(batch,training=False)
        self.val_loss(loss)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        for task,task_loss in losses.items():
            self.val_losses[task](task_loss)
            self.log(f"val/{task}/loss", self.val_losses[task], on_step=False, on_epoch=True, prog_bar=True,
                     metric_attribute=f"val_losses.{task}")

        for task, task_metrics in self.metrics.items():
            for name, metric_func in task_metrics.items():
                eval = metric_func(batch[task])
                self.val_metrics[task][name](eval)
                self.log(f"val/{task}/{name}", self.val_metrics[task][name], batch_size=batch[task].batch_size, on_epoch=True, 
                         prog_bar=True, logger=True, metric_attribute=f"val_metrics.{task}.{name}")

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        for task, metrics in self.val_metrics.items():
            for name, metric in metrics.items():
                current_val = metric.compute()  # get current metric
                self.val_best_metrics[task][name](current_val)  # update best so far metric
                # log metric as a value through `.compute()` method, instead of as a metric object
                # otherwise metric would be reset by lightning after each epoch
                self.log(f"val/{task}/{name}_best", self.val_best_metrics[task][name].compute(), sync_dist=True,
                        prog_bar=(name == "size"))

    def test_step(self, batch, batch_idx):
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data from PyTorch Geometric.
        :param batch_idx: The index of the current batch.
        """
        batch, losses, loss = self.model_step(batch, training=False)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        for task,task_loss in losses.items():
            self.test_losses[task](task_loss)
            self.log(f"test/{task}/loss", self.test_losses[task], on_step=False, on_epoch=True, prog_bar=True, 
                     metric_attribute=f"test_losses.{task}")
        
        for task, task_metrics in self.metrics.items():
            for name, metric_func in task_metrics.items():
                eval = metric_func(batch[task])
                self.test_metrics[task][name](eval)
                self.log(f"test/{task}/{name}", self.test_metrics[task][name], batch_size=batch[task].batch_size, on_epoch=True, 
                         prog_bar=True, logger=True, metric_attribute=f"test_metrics.{task}.{name}")

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

    def on_fit_start(self):
        """Make sure all metrics are on the same device as the model."""
        device = self.device
        self.train_loss.to(device)
        self.val_loss.to(device)
        self.test_loss.to(device)

        for metric_dict in [self.train_losses, self.val_losses, self.test_losses]:
            for metric in metric_dict.values():
                metric.to(device)

        for metrics_group in [self.train_metrics, self.val_metrics, self.test_metrics, self.val_best_metrics]:
            for task_metrics in metrics_group.values():
                for metric in task_metrics.values():
                    metric.to(device)

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
    _ = MultiCOPTModule(None, None, None)
