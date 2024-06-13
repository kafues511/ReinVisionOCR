import cv2
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import warnings
from pytorch_lightning.utilities.warnings import PossibleUserWarning
# num_workers増やせ警告を無視
warnings.simplefilter("ignore", PossibleUserWarning)

from reinlib.utility.rein_torch import get_accelerator

from models.coatnet import CoAtNet
from labelgen.characters_database import CharactersDatabase


__all__ = [
    "my_app",
    "CoAtNetModule",
]


class CoAtNetDataset(Dataset):
    MEAN_NORM = (0.5, )
    STD_NORM = (0.5, )

    def __init__(
        self,
        dataset_directories:list[Path],
        label_to_character_list:list[str],
    ) -> None:
        self.transform = transforms.Compose([
            transforms.Normalize(mean=self.MEAN_NORM, std=self.STD_NORM),
        ])

        self.image_paths:list[tuple[str, int]] = []

        code_n_paths:dict[int, list[Path]] = { ord(character): [] for character in label_to_character_list }
        code_n_label:dict[int, list[Path]] = { ord(character): label for label, character in enumerate(label_to_character_list) }

        for dataset_directory in dataset_directories:
            for path in dataset_directory.glob("*.png"):
                _, _, code = path.stem.split("_")
                code_n_paths[int(code)].append(path)

        for code, paths in code_n_paths.items():
            for path in paths:
                self.image_paths.append((str(path), code_n_label[code]))

        # to tuple
        self.image_paths = tuple(self.image_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index:int) -> tuple[Tensor, int]:
        image_path, label = self.image_paths[index]

        image = read_image(image_path, ImageReadMode.GRAY) / 255.0
        image = self.transform(image)

        return image, label


class CoAtNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_directories:list[str],
        label_to_character_list:list[str],
        batch_size:int,
    ):
        super().__init__()

        self.save_hyperparameters()

    def setup(self, stage:str) -> None:
        self.train_dataset = CoAtNetDataset([Path(dataset_directory) / "train" for dataset_directory in self.hparams.dataset_directories], self.hparams.label_to_character_list)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, pin_memory=True, num_workers=4, shuffle=True, persistent_workers=True)


class CoAtNetModule(pl.LightningModule):
    def __init__(
        self,

        # model
        image_size:tuple[int, int],
        in_channels:int,
        num_blocks:tuple[int, int, int, int, int],
        channels:tuple[int, int, int, int, int],
        num_classes:int,
        block_types:tuple[str, str, str, str],

        # loss
        label_smoothing:float = 0.1,

        # scheduler
        first_cycle_steps:int = 2,
        cycle_mult:float = 1.0,
        max_lr:float = 1.0e-03,
        min_lr:float = 1.0e-05,
        warmup_steps:int = 1,
        gamma:float = 0.5,
        skip_first:bool = True,
    ) -> None:
        super().__init__()

        self.model = CoAtNet(image_size, in_channels, num_blocks, channels, num_classes, block_types)

        self.train_outputs:dict[str, list[Tensor]] = {
            "loss": [],
            "accuracy": [],
        }

        self.save_hyperparameters()

    def forward(self, x:Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch:list[Tensor], batch_idx:int) -> Tensor:
        image, label = batch
        output = self(image)
        loss = F.cross_entropy(output, label, label_smoothing=self.hparams.label_smoothing)
        pred = (torch.argmax(output, dim=1) == label).type(torch.float)
        self.train_outputs["loss"].append(loss)
        self.train_outputs["accuracy"].append(pred)
        return loss

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()

        loss = torch.stack(self.train_outputs["loss"]).mean()
        accuracy = torch.cat(self.train_outputs["accuracy"]).mean()

        self.log("train/loss", loss)
        self.log("train/accuracy", accuracy)
        self.logger.log_metrics({"train/epoch_loss": loss}, self.current_epoch)
        self.logger.log_metrics({"train/epoch_accuracy": accuracy}, self.current_epoch)

        # free up the memory
        self.train_outputs["loss"].clear()
        self.train_outputs["accuracy"].clear()

    def configure_optimizers(self) -> tuple[list[optim.Optimizer], list[optim.lr_scheduler.LRScheduler]]:
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.max_lr,
        )
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.hparams.first_cycle_steps,
            cycle_mult=self.hparams.cycle_mult,
            max_lr=self.hparams.max_lr,
            min_lr=self.hparams.min_lr,
            warmup_steps=self.hparams.warmup_steps,
            gamma=self.hparams.gamma,
        )
        if self.hparams.skip_first:
            scheduler.step()
        return [optimizer], [scheduler]


def my_app(
    dataset_directories:list[str],
    characters_database_path:Optional[str] = None,
    max_epochs:Optional[int] = None,
) -> None:
    """モデルのトレーニング

    Args:
        dataset_directories (list[str]): 学習に使用するデータセットディレクトリのリスト
        characters_database_path (Optional[str], optional): 文字情報のパス、未指定の場合はdataset_directoriesに含まれる情報を利用します. Defaults to None.
        max_epochs (Optional[int], optional): トレーニング回数. Defaults to None.
    """
    torch.set_float32_matmul_precision("high")

    pl.seed_everything(522)

    # トレーニング回数が未指定の場合はデフォルト値を適用します。
    if max_epochs is None:
        max_epochs = 12

    # 文字情報が未指定の場合はデータセットの先頭のデータを利用します。
    if characters_database_path is None:
        characters_database_path = str(Path(dataset_directories[0]) / f"config.{CharactersDatabase.suffix_exclude_dots()}")

    label_to_character_list = CharactersDatabase.load(characters_database_path).label_to_character_list

    num_classes = len(label_to_character_list)

    datamodule = CoAtNetDataModule(dataset_directories, label_to_character_list, 64)

    model = CoAtNetModule(
        (32, 32),
        1,
        (2, 2, 3, 5),
        (64, 96, 192, 384),
        num_classes,
        ("C", "C", "T"),
    )

    logger = TensorBoardLogger(
        save_dir=r"",
        name="log_logs",
        default_hp_metric=False,
    )

    callbacks = [
        LearningRateMonitor(
            log_momentum=False,
        ),
        ModelCheckpoint(
            monitor="train/accuracy",
            filename="checkpoint-epoch={epoch}-accuracy={train/accuracy:.8f}-loss={train/loss:.8f}",
            save_top_k=3,
            mode="max",
            save_last=True,
            auto_insert_metric_name=False,
            save_on_train_epoch_end=True,
        ),
    ]

    trainer = pl.Trainer(
        accelerator=get_accelerator(),
        devices=[0],
        logger=logger,
        callbacks=callbacks,
        max_epochs=max_epochs,
        check_val_every_n_epoch=None,
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
    )
