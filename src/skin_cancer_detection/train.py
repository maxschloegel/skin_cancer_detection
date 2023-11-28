import os
import pathlib

from dotenv import load_dotenv
import hydra
from hydra.utils import get_original_cwd
import mlflow
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from custom_dataset import CustomImageDataset
from model import ImageClassifier
from utils import train_val_split


load_dotenv()


@hydra.main(version_base=None,
            config_path="../../conf",
            config_name="training_config")
def run(cfg: DictConfig):
    hparams = cfg["hparams"]
    logger_cfg = cfg["mlflow_logger"]
    root_dir = pathlib.Path(get_original_cwd())
    data_root = root_dir / "data"
    data_dir = data_root / 'images'
    csv_file = data_root / 'labels.csv'

    transform = nn.Sequential(
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    )

    dataset = CustomImageDataset(data_dir, csv_file, image_column="image_id",
                                 label_column="dx", transform=transform)

    train_dataset, val_dataset = train_val_split(
                                    dataset,
                                    val_split=hparams['val_split'],
                                    seed=hparams['seed'])

    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'],
                              num_workers=hparams['num_workers'],
                              shuffle=hparams["shuffle"])
    val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'],
                            num_workers=hparams["num_workers"])

    mlf_logger = MLFlowLogger(
        experiment_name=logger_cfg["experiment_name"],
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        log_model=logger_cfg["log_model"],
        artifact_location=logger_cfg["artifact_location"]
        )

    model = ImageClassifier(hparams)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(max_epochs=hparams['epochs'],
                         accelerator=accelerator,
                         default_root_dir="checkpoints/",
                         logger=mlf_logger)
    trainer.fit(model, train_loader, val_loader)

    # Apparently the MLFlowLogger does not store the models automatically
    # Maybe this has been fixed already
    os.environ["MLFLOW_RUN_ID"] = trainer.logger.run_id
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = ImageClassifier.load_from_checkpoint(
                    checkpoint_path=best_model_path)

    tf_model = nn.Sequential(transform,
                             model)
    tf_best_model = nn.Sequential(transform,
                                  best_model)

    mlflow.pytorch.log_model(tf_model, "final_model.pt")
    mlflow.pytorch.log_model(tf_best_model, "best_model.pt")


if __name__ == '__main__':
    run()
