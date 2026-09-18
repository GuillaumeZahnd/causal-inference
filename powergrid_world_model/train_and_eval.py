import mlflow
import hydra
import lightning
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf
import torch
from typing import Any, cast

from world_model import NeuralWorldModel, WorldModelDataModule


@hydra.main(version_base=None, config_path="config", config_name="config")
def run_pipeline(cfg: DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))

    torch.set_float32_matmul_precision(cfg.environment.torch_matmul_precision)

    mlflow.set_tracking_uri(cfg.environment.mlflow_tracking_uri)

    mlflow.get_experiment_by_name(cfg.experiment_name)

    mlf_logger = MLFlowLogger(
        experiment_name=cfg.experiment_name,
        run_name=cfg.run_name,
        tracking_uri=cfg.environment.mlflow_tracking_uri,
    )

    params = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    mlf_logger.log_hyperparams(params=params)

    datamodule = WorldModelDataModule(cfg=cfg)
    
    model = NeuralWorldModel(cfg=cfg)

    trainer = lightning.Trainer(
        accelerator=cfg.environment.accelerator,
        max_epochs=cfg.training.nb_epochs,
        logger=mlf_logger,
        enable_checkpointing=True,
    )

    trainer.fit(model=model, datamodule=datamodule)
    
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    run_pipeline()
