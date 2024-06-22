from pathlib import Path
import cv2

import torch
from torch import Tensor, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from torchvision.utils import make_grid

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import warnings
from pytorch_lightning.utilities.warnings import PossibleUserWarning
# num_workers増やせ警告を無視
warnings.simplefilter("ignore", PossibleUserWarning)

from reinlib.utility.rein_torch import get_accelerator

from models.charseg import CharSeg


__all__ = [
    "my_app",
    "CharSegModule",
]


class CharSegDataset(Dataset):
    MEAN_NORM = (0.485, 0.456, 0.406)
    STD_NORM = (0.229, 0.224, 0.225)

    def __init__(
        self,
        dataset_dirs:list[Path],
    ) -> None:
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN_NORM, std=self.STD_NORM),
        ])

        image_paths:list[Path] = []
        label_paths:list[Path] = []
        for dataset_dir in dataset_dirs:
            image_paths.extend((path for path in dataset_dir.glob("*.jpg") if path.stem.startswith("image_")))
            label_paths.extend((path for path in dataset_dir.glob("*.png") if path.stem.startswith("label_")))

        image_paths.sort()
        label_paths.sort()

        self.image_n_label_paths = tuple([
            (str(image_path), str(label_path))
            for image_path, label_path in zip(image_paths, label_paths)
        ])

    def __len__(self) -> int:
        return len(self.image_n_label_paths)

    def __getitem__(self, index:int) -> tuple[Tensor, Tensor]:
        image_path, label_path = self.image_n_label_paths[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = self.transform(image)

        label = read_image(label_path, ImageReadMode.GRAY) / 255.0

        return image, label


class CharSegDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_directories:list[str],
        batch_size:int,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

    def setup(self, stage:str) -> None:
        self.train_dataset = CharSegDataset([Path(dataset_dir) / "train" for dataset_dir in self.hparams.dataset_directories])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, pin_memory=True, num_workers=4, shuffle=True, persistent_workers=True)


class CharSegModule(pl.LightningModule):
    def __init__(
        self,

        # focal loss
        gamma:float = 1.0,

        # optimizer
        lr:float = 1.0e-05,

        # scheduler
        T_max:int = 9,
        eta_min:float = 1.0e-06,
    ) -> None:
        super().__init__()

        self.model = CharSeg()

        self.train_outputs:dict[str, list[Tensor]] = {
            "segmentation": [],
            "loss": [],
            "mse_loss": [],
            "focal_loss": [],
        }

        self.save_hyperparameters()

    def forward(self, x:Tensor) -> Tensor:
        return self.model(x)

    def focal_loss(self, logpt:Tensor) -> Tensor:
        pt = torch.exp(-logpt)
        loss:Tensor = ((1.0 - pt) ** self.hparams.gamma) * logpt
        return loss.mean()

    def training_step(self, batch:list[Tensor], batch_idx:int):
        images, labels = batch
        outputs = torch.clamp(self(images), 0.0, 1.0)
        mse_loss = F.mse_loss(outputs, labels)
        focal_loss = self.focal_loss(mse_loss)
        loss = mse_loss + focal_loss
        self.train_outputs["mse_loss"].append(mse_loss)
        self.train_outputs["focal_loss"].append(focal_loss)
        self.train_outputs["loss"].append(loss)
        if len(self.train_outputs["segmentation"]) == 0:
            self.train_outputs["segmentation"].append(outputs)
        return loss

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()

        mse_loss = torch.stack(self.train_outputs["mse_loss"]).mean()
        focal_loss = torch.stack(self.train_outputs["focal_loss"]).mean()
        loss = torch.stack(self.train_outputs["loss"]).mean()

        self.log("train/mse_loss", mse_loss)
        self.log("train/focal_loss", focal_loss)
        self.log("train/loss", loss)

        self.logger.log_metrics({"train_epoch/mse_loss": mse_loss}, self.current_epoch)
        self.logger.log_metrics({"train_epoch/focal_loss": focal_loss}, self.current_epoch)
        self.logger.log_metrics({"train_epoch/loss": loss}, self.current_epoch)

        self.logger.experiment.add_image(
            "segmentation",
            make_grid(torch.cat(self.train_outputs["segmentation"]), nrow=6),
            self.current_epoch,
            dataformats="CHW",
        )

        # free up the memory
        for key in self.train_outputs.keys():
            self.train_outputs[key].clear()

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
    max_epochs:int = 10,
    batch_size:int = 256,
) -> None:
    """モデルのトレーニング

    Args:
        dataset_directories (list[str]): 学習に使用するデータセットディレクトリのリスト
        max_epochs (int, optional): エポック数. Defaults to 10.
        batch_size (int, optional): バッチサイズ. Defaults to 256.
    """
    torch.set_float32_matmul_precision("high")

    pl.seed_everything(522)

    datamodule = CharSegDataModule(dataset_directories, batch_size)

    model = CharSegModule(T_max=max_epochs - 1)

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
