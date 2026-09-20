import lightning as L
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from hydra.utils import instantiate

from agent import BaseAgent
from data_collector import collect_simulation_data


class TrajectoryDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        feature_cols = [
            "battery_soc",
            "solar_yield",
            "demand_load",
            "spot_price",
            "hour",
            "action_kw",
        ]
        self.x = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.y_soc = torch.tensor(df[["next_battery_soc"]].values, dtype=torch.float32)
        self.y_reward = torch.tensor(df[["reward"]].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y_soc[idx], self.y_reward[idx]


class WorldModelDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.agent: BaseAgent | None = None

    def setup(self, stage: str | None = None):

        if self.agent is None:
            self.agent = instantiate(self.cfg.agent)

        if stage in (None, "fit"):
            train_df = collect_simulation_data(
                agent=self.agent,
                nb_days=self.cfg.data.train_days,
                steps_per_hour=self.cfg.data.steps_per_hour,
                seed=self.cfg.data.train_seed,
            )
            self.train_dataset = TrajectoryDataset(train_df)

        if stage in (None, "test"):
            test_df = collect_simulation_data(
                agent=self.agent,
                nb_days=self.cfg.data.test_days,
                steps_per_hour=self.cfg.data.steps_per_hour,
                seed=self.cfg.data.test_seed,
            )
            self.test_dataset = TrajectoryDataset(test_df)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
        )


class NeuralWorldModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.backbone = nn.Sequential(
            nn.Linear(cfg.model.input_dim, cfg.model.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.model.hidden_dim, cfg.model.hidden_dim),
            nn.ReLU(),
        )
        self.soc_head = nn.Linear(cfg.model.hidden_dim, 1)
        self.reward_head = nn.Linear(cfg.model.hidden_dim, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        return self.soc_head(feat), self.reward_head(feat)

    def training_step(self, batch, batch_idx):
        x, y_soc, y_reward = batch
        pred_soc, pred_reward = self(x)

        loss_soc = self.loss_fn(pred_soc, y_soc)
        loss_reward = self.loss_fn(pred_reward, y_reward)
        loss = loss_soc + loss_reward

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y_soc, y_reward = batch
        pred_soc, pred_reward = self(x)

        soc_mae = torch.abs(pred_soc - y_soc).mean()
        soc_rmse = torch.sqrt(self.loss_fn(pred_soc, y_soc))

        reward_mae = torch.abs(pred_reward - y_reward).mean()
        reward_rmse = torch.sqrt(self.loss_fn(pred_reward, y_reward))

        self.log_dict({
            "test/soc_mae": soc_mae,
            "test/soc_rmse": soc_rmse,
            "test/reward_mae": reward_mae,
            "test/reward_rmse": reward_rmse,
        })

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.model.lr)
