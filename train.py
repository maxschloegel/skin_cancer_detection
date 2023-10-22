import os
import pathlib

import mlflow
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader


from custom_dataset import CustomImageDataset
from utils import train_val_split



class ImageClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        #for key in hparams.keys():
        #    self.hparams[key]=hparams[key]
        #print(self.hparams)
        #self.hparams = hparams
        self.save_hyperparameters(hparams)

        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(hparams['class_labels']))  # Adjust output size for multiple classes

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log('step', self.trainer.current_epoch)
        self.log('step', self.trainer.current_epoch)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = torch.sum(preds == labels.data).item() / labels.size(0)
        self.log('step', self.trainer.current_epoch)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True)

if __name__ == '__main__':
    hparams = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 5,
        'class_labels': ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'],
        'val_split': 0.2,
        'seed': 42,
    }

    data_root = pathlib.Path("data")
    data_dir = data_root / 'images'
    csv_file = data_root / 'labels.csv'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = CustomImageDataset(data_dir, csv_file, image_column="image_id", label_column="dx", transform=transform)

    train_dataset, val_dataset = train_val_split(dataset, val_split=hparams['val_split'], seed=hparams['seed'])

    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'])


    tracking_uri = 'sqlite:///mydb.sqlite'
    mlflow.set_tracking_uri(tracking_uri)
    mlf_logger = MLFlowLogger(
        experiment_name="testing_model_registry_new",
        #tracking_uri='file:./ml-runs',
        tracking_uri=tracking_uri,
        log_model=True,
        artifact_location='mlruns'
        )

    model = ImageClassifier(hparams)

    trainer = pl.Trainer(max_epochs=hparams['epochs'],
                         accelerator= "gpu" if torch.cuda.is_available() else "cpu",
                         default_root_dir="checkpoints/",
                         logger=mlf_logger)
    trainer.fit(model, train_loader, val_loader)

    os.environ["MLFLOW_RUN_ID"] = trainer.logger.run_id
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = ImageClassifier.load_from_checkpoint(checkpoint_path=best_model_path)
    mlflow.pytorch.log_model(model, "final_model.pt")
    mlflow.pytorch.log_model(best_model, "best_model.pt")
