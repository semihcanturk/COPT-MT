from typing import Any, Dict

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pathlib import Path
import shutil

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)

'''
@rank_zero_only
def organize_checkpoint(cfg: DictConfig, trainer: Trainer) -> None:
    """
    Organize trained checkpoint into a standardized location based on tasks.
    Creates: logs/train/checkpoints/{task_names}/last.ckpt
    For color/cliquecover tasks, includes the dim_out value (e.g., color10, color20)
    
    :param cfg: Hydra configuration
    :param trainer: PyTorch Lightning trainer
    """
    from pathlib import Path
    import shutil
    from omegaconf import OmegaConf
    
    # Get the tasks trained on
    tasks = cfg.model.net.get("tasks", [])
    if not tasks:
        log.warning("No tasks found in config, skipping checkpoint organization")
        return
    
    # Get dim_out to handle color/cliquecover tasks
    dim_out = cfg.model.net.get("dim_out", {})
    
    # Create task names with dim_out for color/cliquecover
    task_names = []
    for task in sorted(tasks):
        if task in ["color", "cliquecover"] and task in dim_out:
            # Append the number of colors/cliques to the task name
            task_names.append(f"{task}{dim_out[task]}")
        else:
            task_names.append(task)
    
    # Create task identifier (e.g., "maxcut" or "color10_mis" or "color10_color20")
    task_name = "_".join(task_names)
    
    # Determine source checkpoint
    if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback:
        # Get the last checkpoint path
        ckpt_callback = trainer.checkpoint_callback
        source_ckpt = Path(ckpt_callback.dirpath) / "last.ckpt"
        
        if not source_ckpt.exists():
            log.warning(f"Checkpoint not found at {source_ckpt}, skipping organization")
            return
        
        # Create organized checkpoint directory
        base_dir = Path(cfg.paths.get("checkpoint_dir", "logs/train/checkpoints"))
        organized_dir = base_dir / task_name
        organized_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy checkpoint
        dest_ckpt = organized_dir / "last.ckpt"
        shutil.copy2(source_ckpt, dest_ckpt)
        
        log.info(f"Organized checkpoint: {dest_ckpt}")
        
        # Also copy the best checkpoint if it exists and is different
        if ckpt_callback.best_model_path and Path(ckpt_callback.best_model_path).exists():
            best_source = Path(ckpt_callback.best_model_path)
            if best_source != source_ckpt:
                best_dest = organized_dir / "best.ckpt"
                shutil.copy2(best_source, best_dest)
                log.info(f"Organized best checkpoint: {best_dest}")
        
        # Create a metadata file with run info
        metadata = {
            "tasks": OmegaConf.to_container(tasks, resolve=True),
            "dim_out": OmegaConf.to_container(dim_out, resolve=True),
            "timestamp": cfg.paths.output_dir.split("/")[-1],
            "seed": cfg.get("seed"),
            "tags": OmegaConf.to_container(cfg.get("tags", []), resolve=True),
            "checkpoint_path": str(dest_ckpt),
            "best_checkpoint_path": str(organized_dir / "best.ckpt") if ckpt_callback.best_model_path else None,
        }
        
        import json
        metadata_file = organized_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        log.info(f"Saved metadata: {metadata_file}")
    else:
        log.warning("No checkpoint callback found, skipping checkpoint organization")
'''

@rank_zero_only
def organize_checkpoint(cfg: DictConfig, trainer: Trainer) -> None:
    """
    Organize trained checkpoint into a standardized location based on tasks.
    Creates: logs/train/checkpoints/{task_names}/last.ckpt
    For color/cliquecover tasks, includes the dim_out value (e.g., color10, color20)
    Saves both last.ckpt and best.ckpt based on the monitored metric.
    
    :param cfg: Hydra configuration
    :param trainer: PyTorch Lightning trainer
    """
    from pathlib import Path
    import shutil
    from omegaconf import OmegaConf
    
    # Get the tasks trained on
    tasks = cfg.model.net.get("tasks", [])
    if not tasks:
        log.warning("No tasks found in config, skipping checkpoint organization")
        return
    
    # Get dim_out to handle color/cliquecover tasks
    dim_out = cfg.model.net.get("dim_out", {})
    
    # Create task names with dim_out for color/cliquecover
    task_names = []
    for task in sorted(tasks):
        if task in ["color", "cliquecover"] and task in dim_out:
            # Append the number of colors/cliques to the task name
            task_names.append(f"{task}{dim_out[task]}")
        else:
            task_names.append(task)
    
    # Create task identifier (e.g., "maxcut" or "color10_mis" or "color10_color20")
    task_name = "_".join(task_names)
    
    # Determine source checkpoint
    if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback:
        ckpt_callback = trainer.checkpoint_callback
        
        # Create organized checkpoint directory
        base_dir = Path(cfg.paths.get("checkpoint_dir", "logs/train/checkpoints"))
        organized_dir = base_dir / task_name
        organized_dir.mkdir(parents=True, exist_ok=True)
        
        # Save last checkpoint
        source_last = Path(ckpt_callback.dirpath) / "last.ckpt"
        if source_last.exists():
            dest_last = organized_dir / "last.ckpt"
            shutil.copy2(source_last, dest_last)
            log.info(f"Organized checkpoint: {dest_last}")
        else:
            log.warning(f"Last checkpoint not found at {source_last}")
            dest_last = None
        
        # Save best checkpoint (based on monitored metric)
        best_ckpt_path = None
        if ckpt_callback.best_model_path and Path(ckpt_callback.best_model_path).exists():
            best_source = Path(ckpt_callback.best_model_path)
            dest_best = organized_dir / "best.ckpt"
            shutil.copy2(best_source, dest_best)
            log.info(f"Organized best checkpoint: {dest_best}")
            best_ckpt_path = str(dest_best)
        else:
            log.warning("Best checkpoint not found")
        
        # Collect best metrics from the model's val_best_metrics
        best_metrics = {}
        if hasattr(trainer.lightning_module, 'val_best_metrics'):
            for task, task_metrics in trainer.lightning_module.val_best_metrics.items():
                best_metrics[task] = {}
                for metric_name, metric_obj in task_metrics.items():
                    try:
                        best_metrics[task][metric_name] = float(metric_obj.compute().item())
                    except:
                        best_metrics[task][metric_name] = None
        
        # Create metadata file with run info and best metrics
        metadata = {
            "tasks": OmegaConf.to_container(tasks, resolve=True),
            "dim_out": OmegaConf.to_container(dim_out, resolve=True),
            "timestamp": cfg.paths.output_dir.split("/")[-1],
            "seed": cfg.get("seed"),
            "tags": OmegaConf.to_container(cfg.get("tags", []), resolve=True),
            "checkpoint_path": str(dest_last) if dest_last else None,
            "best_checkpoint_path": best_ckpt_path,
            "monitored_metric": ckpt_callback.monitor if hasattr(ckpt_callback, 'monitor') else None,
            "best_metrics": best_metrics,
            "trainer_config": {
                "max_epochs": cfg.trainer.get("max_epochs"),
                "gradient_clip_val": cfg.trainer.get("gradient_clip_val"),
            }
        }
        
        import json
        metadata_file = organized_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        log.info(f"Saved metadata: {metadata_file}")
    else:
        log.warning("No checkpoint callback found, skipping checkpoint organization")