import monai as mn
import torch
import numpy as np

class Transform4ClassifierBase:
    def __init__(self, IMG_SIZE, CLASSES):
        """
        Initializes a set of data transformations for image classification.

        Args:
            IMG_SIZE (int): Desired spatial size for input images.
        """
        # Define the training data transformations (no LoadImageD since we are using NumPy arrays)
        self.train = mn.transforms.Compose([
            # Assume that "img" key already contains the NumPy array
            mn.transforms.Lambdad(keys="img", func=lambda x: np.moveaxis(x, -1, 0) if x.shape[-1] == 1 else x),
            # mn.transforms.SqueezeDimd(keys="img", dim=3, allow_missing_keys=False),
            mn.transforms.HistogramNormalized(keys="img"),
            mn.transforms.ScaleIntensityRangePercentilesD(keys="img", lower=0, upper=100, b_min=0, b_max=1, clip=True),
            mn.transforms.ResizeD(keys='img', spatial_size=IMG_SIZE, size_mode="longest", mode="bilinear", align_corners=False),
            mn.transforms.SpatialPadd(keys="img", spatial_size=(IMG_SIZE, IMG_SIZE), mode="constant", constant_values=0),
            mn.transforms.RandFlipD(keys="img", spatial_axis=0, prob=0.2).set_random_state(seed=2024),
            mn.transforms.RandFlipD(keys="img", spatial_axis=1, prob=0.2).set_random_state(seed=2024),
            mn.transforms.RandGaussianNoiseD(keys="img", mean=0.0, std=0.3, prob=0.5).set_random_state(seed=2024),
            mn.transforms.RandAffineD(keys="img", mode="bilinear", prob=0.5, rotate_range=0.4, scale_range=0.1, translate_range=IMG_SIZE//20, padding_mode="border").set_random_state(seed=2024),
            mn.transforms.ToTensord(keys="img", dtype=torch.float),
            mn.transforms.ToTensord(keys=[*CLASSES], dtype=torch.float),  # Convert class labels to tensors
            mn.transforms.EnsureChannelFirstd(keys=[*CLASSES], channel_dim='no_channel'),  # Ensure channels first for class labels
            mn.transforms.ConcatItemsd(keys=[*CLASSES], name='label'),  # Combine the class labels
            mn.transforms.IdentityD(keys="dicom", allow_missing_keys=True),  # Retain dicom key unchanged
            mn.transforms.SelectItemsd(keys=["img", "label", "dicom"])  # Select only "img" and "label" for further processing
        ])

        # Define the validation data transformations
        self.val = mn.transforms.Compose([
            # Assume that "img" key already contains the NumPy array
            mn.transforms.Lambdad(keys="img", func=lambda x: np.moveaxis(x, -1, 0) if x.shape[-1] == 1 else x),
            # mn.transforms.SqueezeDimd(keys="img", dim=3, allow_missing_keys=False),
            mn.transforms.HistogramNormalized(keys="img"),
            mn.transforms.ScaleIntensityRangePercentilesD(keys="img", lower=0, upper=100, b_min=0, b_max=1, clip=True),
            mn.transforms.ResizeD(keys='img', spatial_size=IMG_SIZE, size_mode="longest", mode="bilinear", align_corners=False),
            mn.transforms.SpatialPadd(keys="img", spatial_size=(IMG_SIZE, IMG_SIZE), mode="constant", constant_values=0),
            mn.transforms.ToTensord(keys="img", dtype=torch.float),
            mn.transforms.ToTensord(keys=[*CLASSES], dtype=torch.float),  # Convert class labels to tensors
            mn.transforms.EnsureChannelFirstd(keys=[*CLASSES], channel_dim='no_channel'),  # Ensure channels first for class labels
            mn.transforms.ConcatItemsd(keys=[*CLASSES], name='label'),  # Combine the class labels
            mn.transforms.IdentityD(keys="dicom", allow_missing_keys=True),  # Retain dicom key unchanged
            mn.transforms.SelectItemsd(keys=["img", "label", "dicom"])  # Select only "img" and "label" for further processing
        ])

        self.test = mn.transforms.Compose([
            # Assume that "img" key already contains the NumPy array
            mn.transforms.Lambdad(keys="img", func=lambda x: np.moveaxis(x, -1, 0) if x.shape[-1] == 1 else x),
            # mn.transforms.SqueezeDimd(keys="img", dim=3, allow_missing_keys=False),
            mn.transforms.HistogramNormalized(keys="img"),
            mn.transforms.ScaleIntensityRangePercentilesD(keys="img", lower=0, upper=100, b_min=0, b_max=1, clip=True),
            mn.transforms.ResizeD(keys='img', spatial_size=IMG_SIZE, size_mode="longest", mode="bilinear", align_corners=False),
            mn.transforms.SpatialPadd(keys="img", spatial_size=(IMG_SIZE, IMG_SIZE), mode="constant", constant_values=0),
            mn.transforms.ToTensord(keys="img", dtype=torch.float),
            mn.transforms.ToTensord(keys=[*CLASSES], dtype=torch.float),  # Convert class labels to tensors
            mn.transforms.EnsureChannelFirstd(keys=[*CLASSES], channel_dim='no_channel'),  # Ensure channels first for class labels
            mn.transforms.ConcatItemsd(keys=[*CLASSES], name='label'),  # Combine the class labels
            mn.transforms.IdentityD(keys="dicom", allow_missing_keys=True),  # Retain dicom key unchanged
            mn.transforms.SelectItemsd(keys=["img", "label", "dicom"])  # Select only "img" and "label" for further processing
        ])