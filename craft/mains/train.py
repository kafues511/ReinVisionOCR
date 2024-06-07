import cv2
from pathlib import Path
import pickle
import numpy as np

import torch
from torch import Tensor
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import warnings
from pytorch_lightning.utilities.warnings import PossibleUserWarning
# num_workers増やせ警告を無視
warnings.simplefilter("ignore", PossibleUserWarning)

from reinlib.types.rein_size2d import Size2D
from reinlib.types.rein_bounding_box import BoundingBox
from reinlib.utility.rein_torch import get_accelerator

from models.craft import CRAFT
from models.gaussian import GaussianGenerator


__all__ = [
    "my_app",
    "CRAFTModule",
]


class CRAFTDataset(Dataset):
    MEAN_NORM = (0.485, 0.456, 0.406)
    STD_NORM = (0.229, 0.224, 0.225)

    def __init__(
        self,
        dataset_directories:list[Path],
        dratios:list[float],
        version:int = 0,
    ) -> None:
        self.transform = transforms.Compose([
            transforms.Normalize(mean=self.MEAN_NORM, std=self.STD_NORM),
        ])

        self.gaussians:dict[float, GaussianGenerator] = {}
        self.image_paths:list[str] = []
        self.bboxes_paths:list[tuple[str, float]] = []

        for dataset_directory, dratio in zip(dataset_directories, dratios):
            self.image_paths.extend([str(path) for path in dataset_directory.glob("*.jpg") if path.stem.startswith("image_")])
            self.bboxes_paths.extend([(str(path), dratio) for path in dataset_directory.glob("*.pkl") if path.stem.startswith("bboxes_")])
            if dratio not in list(self.gaussians.keys()):
                self.gaussians[dratio] = GaussianGenerator(distance_ratio=dratio)

        assert len(self.image_paths) == len(self.bboxes_paths), "not match length."

        self.version = version
        self.image_paths = tuple(self.image_paths)
        self.bboxes_paths = tuple(self.bboxes_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index:int) -> list[Tensor]:
        image_path = self.image_paths[index]
        bboxes_path, dratio = self.bboxes_paths[index]

        image = read_image(image_path, ImageReadMode.RGB) / 255.0
        image = self.transform(image)

        with open(str(bboxes_path), mode="rb") as f:
            bboxes:list[BoundingBox] = pickle.load(f)

        c, h, w = image.shape

        full_size = Size2D(w, h)
        half_size = full_size//2

        # NOTE: 文字領域をハーフにしてガウス生成するより、フルサイズなガウスをハーフにリサイズした方が高品質
        if self.version == 0:
            label = self.gaussians[dratio](full_size, bboxes) / 255.0
            label = cv2.resize(label, half_size.wh, interpolation=cv2.INTER_NEAREST).astype(np.float32)
        elif self.version == 1:
            label = self.gaussians[dratio](full_size, bboxes) / 255.0
            label = cv2.resize(label, half_size.wh, interpolation=cv2.INTER_AREA).astype(np.float32)
        elif self.version == 2:
            label = self.gaussians[dratio](full_size, bboxes) / 255.0
            label = cv2.resize(label, half_size.wh, interpolation=cv2.INTER_LINEAR).astype(np.float32)
        elif self.version == 3:
            bboxes = [bbox//2 for bbox in bboxes]
            label = (self.gaussians[dratio](full_size, bboxes) / 255.0).astype(np.float32)

        label = to_tensor(label)

        return [image, label]


class CRAFTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_directories:list[str],
        dratios:list[float],
        batch_size:int,
    ) -> None:
        assert len(dataset_directories) == len(dratios), "not match length."

        super().__init__()

        self.save_hyperparameters()

    def setup(self, stage:str) -> None:
        self.train_dataset = CRAFTDataset([Path(dataset_dir) / "train" for dataset_dir in self.hparams.dataset_directories], self.hparams.dratios)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, pin_memory=True, num_workers=4, shuffle=True, persistent_workers=True)


class CRAFTModule(pl.LightningModule):
    def __init__(
        self,

        # optimizer
        lr:float=1.0e-03,

        # scheduler
        T_max:int=9,
        eta_min:float=1.0e-05,
    ):
        super().__init__()

        self.model = CRAFT()

        self.train_outputs:dict[str, list[Tensor]] = {
            "loss": [],
            "mse_loss": [],
        }

        self.save_hyperparameters()

    def forward(self, x:Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch:list[Tensor], batch_idx:int):
        images, labels = batch
        outputs = self(images)
        mse_loss = F.mse_loss(outputs, labels)
        loss = mse_loss
        self.train_outputs["mse_loss"].append(mse_loss)
        self.train_outputs["loss"].append(loss)
        return loss

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()

        mse_loss = torch.stack(self.train_outputs["mse_loss"]).mean()
        loss = torch.stack(self.train_outputs["loss"]).mean()

        self.log("train/mse_loss", mse_loss)
        self.log("train/loss", loss)

        self.logger.log_metrics({"train_epoch/mse_loss": mse_loss}, self.current_epoch)
        self.logger.log_metrics({"train_epoch/loss": loss}, self.current_epoch)

        # free up the memory
        self.train_outputs["mse_loss"].clear()
        self.train_outputs["loss"].clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.T_max,
            eta_min=self.hparams.eta_min,
        )
        return [optimizer], [scheduler]


def my_app(
    dataset_directories:list[str],
    dratios:list[float],
    max_epochs:int = 10,
    batch_size:int = 8,
) -> None:
    """モデルのトレーニング

    Args:
        dataset_directories (list[str]): 学習に使用するデータセットディレクトリのリスト
        max_epochs (int, optional): エポック数. Defaults to 10.
        batch_size (int, optional): バッチサイズ. Defaults to 8.
    """
    torch.set_float32_matmul_precision("high")

    pl.seed_everything(522)

    datamodule = CRAFTDataModule(dataset_directories, dratios, batch_size)

    model = CRAFTModule(T_max=max_epochs - 1)

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
            monitor="train/loss",
            filename="checkpoint-epoch={epoch}-loss={train/loss:.8f}",
            save_top_k=3,
            mode="min",
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
