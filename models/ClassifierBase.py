import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel
import lightning as pl
import timm
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel
import lightning as pl
import timm
import pandas as pd  # Import pandas for DataFrame creation
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import torch.nn.functional as F
import numpy as np 

class Classifier(pl.LightningModule):
    """
    Example of a classification model using PyTorch Lightning.

    Args:
        TIMM_MODEL (str): Pre-trained model architecture (default: "convnext_base.fb_in22k_ft_in1k").
        LEARNING_RATE (float): Learning rate for the optimizer.
        BATCH_SIZE (int): Batch size for training and validation.
        use_ema (bool): Whether to use exponential moving average (EMA) during training.
    """
    def __init__(self, TIMM_MODEL='convnext_base.fb_in22k_ft_in1k', LEARNING_RATE=1e-5, BATCH_SIZE=32, use_ema=False, num_classes=14, task_name = None, set_num = None):
        super().__init__()
        self.set_num = -1
        self.task_name =  task_name
        self.use_ema = use_ema
        self.TIMM_MODEL = TIMM_MODEL
        self.LEARNING_RATE = LEARNING_RATE
        self.BATCH_SIZE = BATCH_SIZE
        self.model = timm.create_model(self.TIMM_MODEL, pretrained=True, in_chans=1, num_classes=num_classes)
        self.ema = AveragedModel(self.model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
        self.val_logits = []
        self.val_labels = []
    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input data (image).

        Returns:
            torch.Tensor: Embedding (output) from the model.
        """

        embedding = self.model(x)

        return embedding
    def forward_features(self, x):
        """
        Extract features before final classification head.
        This uses the timm model's built-in feature extractor.
        """
        return self.model.forward_features(x)

    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns:
            torch.optim.Optimizer: Adam optimizer with model parameters.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.LEARNING_RATE)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        """
        Defines the training logic for one batch.

        Args:
            train_batch (dict): Training batch containing 'img' and 'label'.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Training loss.
        """
        x, y = train_batch['img'], train_batch['label']
        logits = self(x)
        y = y.float()
        criterion = nn.BCEWithLogitsLoss(reduction="mean")
        loss = criterion(logits, y)
        self.log('train_loss', loss.item(), on_epoch=True, on_step=False, prog_bar=False, batch_size=self.BATCH_SIZE)
        return loss

    def validation_step(self, val_batch, batch_idx):
        """
        Defines the validation logic for one batch.
    
        Args:
            val_batch (dict): Validation batch containing 'img' and 'label'.
            batch_idx (int): Batch index.
    
        Returns:
            dict: Dictionary containing validation loss and metrics.
        """
        x, y = val_batch['img'], val_batch['label']
        
        # Get the logits using EMA or the regular model
        if self.use_ema:
            logits = self.ema(x)
        else:
            logits = self(x)
    
        y = y.float()
        criterion = nn.BCEWithLogitsLoss(reduction="mean")
        
        # Calculate loss
        loss = criterion(logits, y)
    
        # Log validation loss
        self.log('valid_loss', loss.item(), on_epoch=True, on_step=True, prog_bar=False, batch_size=self.BATCH_SIZE)
    
        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(logits)
        
        # Binarize predictions (threshold = 0.5)
        preds = (probs >= 0.5).float()
    
        # Convert to float32 for compatibility with roc_auc_score
        y_cpu = y.float().cpu().numpy()  # Ensure it's a regular tensor and cast to numpy
        probs_cpu = probs.float().cpu().numpy()  # Ensure it's a regular tensor and cast to numpy
    
        # Calculate AUROC
        if len(torch.unique(y)) > 1:  # Ensure that there are both positive and negative samples
            auroc = roc_auc_score(y_cpu, probs_cpu)
        else:
            auroc = float('nan')  # Handle edge case where AUROC is not defined
    
        # Calculate accuracy
        accuracy = accuracy_score(y.cpu().numpy(), preds.cpu().numpy())
    
        # Calculate confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y.cpu().numpy(), preds.cpu().numpy(), labels=[0, 1]).ravel()
        
        # True Positive Rate (TPR) and True Negative Rate (TNR)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity/Recall
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
    
        # Log AUROC, accuracy, TPR, and TNR
        self.log('valid_auroc', auroc, on_epoch=True, on_step=False, prog_bar=True, batch_size=self.BATCH_SIZE)
        self.log('valid_accuracy', accuracy, on_epoch=True, on_step=False, prog_bar=True, batch_size=self.BATCH_SIZE)
        self.log('valid_tpr', tpr, on_epoch=True, on_step=False, prog_bar=False, batch_size=self.BATCH_SIZE)
        self.log('valid_tnr', tnr, on_epoch=True, on_step=False, prog_bar=False, batch_size=self.BATCH_SIZE)
        self.val_logits = self.val_logits + (probs_cpu.tolist())
        self.val_labels = self.val_labels + (y_cpu.tolist())
        return loss

    def predict_step(self, val_batch, batch_idx):
        """
        Computes predictions for validation data.

        Args:
            val_batch (dict): Validation batch containing 'img' and 'label'.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Validation loss.
        """
        x, y = val_batch['img'], val_batch['label']
        if self.use_ema:
            logits = self.ema(x)
        else:
            logits = self(x)
        y = y.float()
        probs = torch.sigmoid(logits)
        return {'dicom': val_batch['dicom'],
                     'Logit': torch.squeeze(logits).cpu().numpy(),
                     'Prob': torch.squeeze(probs).cpu().numpy(),
                     'Label': torch.squeeze(y).cpu().numpy()}

    def optimizer_step(self, *args, **kwargs):
        """
        Updates optimizer parameters.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update_parameters(self.model)
    def on_validation_epoch_end(self):
        """
        Compute metrics at the end of the validation epoch.
        """
       
        # Concatenate all stored logits and labels
        logits = self.val_logits
        labels = self.val_labels
    
        # Convert tensors to numpy for sklearn
        labels_np = np.array(labels)
        probs_np = np.array(logits)
    
        # Calculate AUROC for the whole validation set
        
        auroc = roc_auc_score(labels, logits)

    
        # Log the whole-set AUROC
        self.log('epoch_auroc', auroc, on_epoch=True, prog_bar=True)
    
        # Clear the stored logits and labels for the next validation epoch
        self.val_logits = []
        self.val_labels = []


class Classifier_multi_label(pl.LightningModule):
    def __init__(self, TIMM_MODEL='convnext_base.fb_in22k_ft_in1k', LEARNING_RATE=1e-5, BATCH_SIZE=32, use_ema=False, num_classes=3):
        super().__init__()

        self.use_ema = use_ema
        self.TIMM_MODEL = TIMM_MODEL
        self.LEARNING_RATE = LEARNING_RATE
        self.BATCH_SIZE = BATCH_SIZE
        self.num_classes = num_classes

        self.model = timm.create_model(self.TIMM_MODEL, pretrained=True, in_chans=1, num_classes=num_classes)
        self.ema = AveragedModel(self.model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
        
        self.val_logits = []
        self.val_labels = []
        self.best_auroc = -1
        self.best_epoch_metrics = None

    def forward(self, x):
        return self.model(x)

    def forward_features(self, x):
        return self.model.forward_features(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.LEARNING_RATE)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch['img'], train_batch['label']
        logits = self(x)
        y = torch.as_tensor(y).long()
        loss = nn.CrossEntropyLoss()(logits, y.long())
        self.log('train_loss', loss.item(), on_epoch=True, on_step=False, prog_bar=False, batch_size=self.BATCH_SIZE)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch['img'], val_batch['label']
        logits = self.ema(x) if self.use_ema else self(x)
        y = torch.as_tensor(y).long()
        loss = nn.CrossEntropyLoss()(logits, y.long())
        self.log('valid_loss', loss.item(), on_epoch=True, on_step=True, prog_bar=False, batch_size=self.BATCH_SIZE)

        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        y_np = y.cpu().numpy()
        probs_np = probs.cpu().numpy()
        preds_np = preds.cpu().numpy()

        # 多類別 AUROC（macro average）
        try:
            auroc = roc_auc_score(y_np, probs_np, multi_class='ovr', average='macro')
        except:
            auroc = float('nan')

        acc = accuracy_score(y_np, preds_np)

        self.log('valid_accuracy', acc, on_epoch=True, prog_bar=True)
        self.log('valid_auroc', auroc, on_epoch=True, prog_bar=True)

        self.val_logits += probs_np.tolist()
        self.val_labels += y_np.tolist()
        return loss

    def predict_step(self, val_batch, batch_idx):
        x, y = val_batch['img'], val_batch['label']
        logits = self.ema(x) if self.use_ema else self(x)

        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        return {
            'dicom': val_batch['dicom'],
            'Logit': logits.cpu().numpy(),
            'Prob': probs.cpu().numpy(),
            'Label': y.cpu().numpy(),
            'Pred': preds.cpu().numpy()
        }

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.use_ema:
            self.ema.update_parameters(self.model)

    def on_validation_epoch_end(self):
        if len(self.val_logits) > 0:
            labels_np = np.array(self.val_labels)
            probs_np = np.array(self.val_logits)

            try:
                auroc = roc_auc_score(labels_np, probs_np, multi_class='ovr', average='macro')
            except:
                auroc = float('nan')

            self.log('epoch_auroc', auroc, on_epoch=True, prog_bar=True)
            if auroc > self.best_auroc:
                self.best_auroc = auroc
                self.best_epoch_metrics = {
                    'epoch': self.current_epoch,
                    'auroc': auroc,
                    'tpr': tpr,
                    'tnr': tnr
                }
            self.val_logits = []
            self.val_labels = []
    def on_train_end(self):
        if self.best_epoch_metrics:
            os.makedirs(f'sub_race_epoch_metrics/IND/', exist_ok=True)
            csv_path = f'sub_race_epoch_metrics/IND/all_tasks_{self.set_num}.csv'
   
            row = {
                'task': self.task_name,
                'epoch': self.best_epoch_metrics['epoch'],
                'auroc': self.best_epoch_metrics['auroc'],
                'tpr': self.best_epoch_metrics['tpr'],
                'tnr': self.best_epoch_metrics['tnr']
            }
    
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            else:
                df = pd.DataFrame([row])
    
            df.to_csv(csv_path, index=False)
            print(f"[INFO] Appended best AUROC metrics to: {csv_path}")



