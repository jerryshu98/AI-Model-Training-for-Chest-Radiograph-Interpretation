import os
import sys
import argparse
import subprocess
import torch
import pandas as pd
import numpy as np
from glob import glob
from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch.callbacks import RichProgressBar
import monai as mn
from transforms import Transform4ClassifierBase
from models import ClassifierBase
from utility import load_data_from_pickle, merge_image, get_data_dict

def main():
    parser = argparse.ArgumentParser(description="Run inference on a trained model.")
    parser.add_argument("label_index", type=str, help="Index of the label to be used for training.")
    parser.add_argument("model_path", type=str, help="Path to the pretrained TIMM model")
    args = parser.parse_args()

    SEED = 5566
    pl.seed_everything(SEED)
    torch.set_float32_matmul_precision('medium')

    # Parse args
    label_index = args.label_index
    model_path = args.model_path
    PATHOLOGIES = [f'Label_{label_index}']

    # Paths
    INPUT = f"./dataset/train_{PATHOLOGIES[0]}.csv"
    PROJECT = f'CNN_{PATHOLOGIES[0]}'
    TEST_NAME = f'{PROJECT}_{model_path.split("/")[-1]}'
    MONAI_CACHE_DIR = f'./cache/{TEST_NAME}'
    WEIGHT_PATH = glob(f'./weights/{TEST_NAME}/*.ckpt')[0]
    BATCH_SIZE = 64

    # GPU config
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    # Clean MONAI cache
    if os.path.exists(MONAI_CACHE_DIR):
        subprocess.call(['rm', '-rf', MONAI_CACHE_DIR])
        print(f"MONAI cache directory {MONAI_CACHE_DIR} removed!")

    # Load image info and dataset
    image_df = load_data_from_pickle("./dataset/study_image_dicom.pkl")
    df = pd.read_csv(INPUT)
    df = merge_image(df)
    eval_dict = get_data_dict(df)

    # Transform and dataloader
    T = Transform4ClassifierBase.Transform4ClassifierBase(224, PATHOLOGIES)
    eval_ds = mn.data.PersistentDataset(data=eval_dict, transform=T.test, cache_dir=MONAI_CACHE_DIR)
    eval_dl = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=False)

    # Load model
    model = ClassifierBase.Classifier(TIMM_MODEL=model_path, BATCH_SIZE=BATCH_SIZE, use_ema=True, num_classes=1)
    ckpt = torch.load(WEIGHT_PATH)['state_dict']
    model.load_state_dict(ckpt)

    # Run inference
    trainer = pl.Trainer(callbacks=[RichProgressBar()])
    predictions = trainer.predict(model, dataloaders=eval_dl)

    # Format prediction results
    df_result = pd.DataFrame()
    for p in predictions:
        dicom = pd.DataFrame(p['dicom'], columns=['dicom'])
        logit = pd.DataFrame(p['Logit'], columns=[f'{i}_Logit' for i in PATHOLOGIES])
        prob = pd.DataFrame(p['Prob'], columns=[f'{i}_Prob' for i in PATHOLOGIES])
        df_temp = pd.concat([dicom, logit, prob], axis=1)
        df_result = pd.concat([df_result, df_temp], axis=0)

    # Merge with ground truth
    df_result = pd.merge(df_result, df[['dicom', PATHOLOGIES[0]]], on='dicom', how='left')
    predicted_labels = np.where(df_result[f'{PATHOLOGIES[0]}_Prob'] >= 0.5, 1, 0)
    accuracy = np.mean(predicted_labels == df_result[PATHOLOGIES[0]])
    print(f'Accuracy: {accuracy:.4f}')

    # Save
    output_path = f"./dataset/train_pre/pre_data_{TEST_NAME}.csv"
    df_result.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
