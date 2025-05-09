import sys
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
import monai as mn
from transforms import Transform4ClassifierBase
from models import ClassifierBase
from utility import setup_gpu, get_data_dict_part, load_data_from_pickle


import argparse


def main():
    parser = argparse.ArgumentParser(description="Train a model with specified label and model path.")
    parser.add_argument("label_index", type=str, help="Index of the label to be used for training.")
    parser.add_argument("model_path", type=str, help="Path to the pretrained TIMM model")
    args = parser.parse_args()

    label_index = args.label_index
    model_path = args.model_path
    print("Usage: python train.py <label_index> <model_path>")
    sys.exit(1)

    # Parse arguments
    label_index = argv[1]
    model_path = argv[2]

    # Configuration
    PATHOLOGIES = [f'Label_{label_index}']
    PROJECT = f'Models_AP_{PATHOLOGIES[0]}'
    TEST_NAME = f'j_{PROJECT}_{model_path.split("/")[-1]}'
    WEIGHT_PATH = f'./weights/{TEST_NAME}'

    # Data Preparation
    image_df = load_data_from_pickle("/mnt/new_usb/jerry_backup/harm_label_weight/study_image_dicom.pkl")
    train_df = pd.read_csv(f"./dataset/train_{PATHOLOGIES[0]}.csv")
    val_df = pd.read_csv(f"./dataset/val_{PATHOLOGIES[0]}.csv")

    # Merge with metadata and filter
    train_df = train_df.merge(image_df, on="study_id", how="right")
    val_df = val_df.merge(image_df, on="study_id", how="right")

    # Data Dictionaries
    train_dict = get_data_dict_part(train_df)
    val_dict = get_data_dict_part(val_df)

    # Transforms
    T = Transform4ClassifierBase.Transform4ClassifierBase(224, PATHOLOGIES)
    train_transforms, val_transforms = T.train, T.val

    # Dataset and Dataloader
    train_ds = mn.data.PersistentDataset(data=train_dict, transform=train_transforms)
    val_ds = mn.data.PersistentDataset(data=val_dict, transform=val_transforms)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

    # Model Initialization
    model = ClassifierBase.Classifier(TIMM_MODEL=model_path, LEARNING_RATE=1e-5, BATCH_SIZE=64, num_classes=1)

    # Trainer Setup
    trainer = pl.Trainer(
        max_epochs=500,
        accelerator="gpu",
        devices=1,
        logger=WandbLogger(project=PROJECT, name=TEST_NAME)
    )

    # Start Training
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    print(f"Training for {TEST_NAME} complete.")


if __name__ == "__main__":
    main()
