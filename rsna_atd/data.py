
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from monai.transforms import Pad
from monai.transforms.transform import Transform
import rsna_atd.config as config
from monai.transforms import LoadImage

def decode_image_and_label(image_path, image_size=None):
    if image_size is None:
        image_size = config.IMAGE_SIZE
    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size, Image.BILINEAR)
    image = transforms.ToTensor()(image)
    return image


class CustomRandomApply(torch.nn.Module):
    """Apply randomly a list of transformations. Each transformation is applied with a probability of p.

    Args:
        transforms (sequence or torch.nn.Module): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, img):
        for t in self.transforms:
            if self.p < torch.rand(1):
                img = t(img)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    p={self.p}"
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, image_size = None):
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = decode_image_and_label(image_path, self.image_size)
        return image, label
    
def apply_2d_augmentation(images, labels, p=0.1):
    augmenter = CustomRandomApply(p=p,
                                  transforms=[
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomRotation(degrees=(0, 30)),
        transforms.ElasticTransform(alpha=50.0),
        transforms.RandomResizedCrop(size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
                                     scale=(0.9, 0.99)),
        transforms.RandomPosterize(bits=8, p=0.7),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5)
    ])
    
    augmented_images = torch.stack([augmenter(img) for img in images])
    return augmented_images, labels

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(config.IMAGE_SIZE, Image.BILINEAR)
    image = transforms.ToTensor()(image)
    return image

def max_shape_pad(tensor_list):
    """Pad tensors to the maximum shape in the list.
    
    Args:
        tensor_list (list): List of tensors to pad.

    Returns:
        cat_tensors (torch.Tensor): Padded tensors concatenated along the first dimension.

    # Example usage
    ```python
    tensor_list = [torch.tensor([[1, 2, 3, 4, 5 ,6]]), torch.tensor([[3, 4, 5]])]
    padded_tensors = max_shape_pad(tensor_list)
    print(padded_tensors)
    for tensor in padded_tensors:
    print(tensor)
    ```
    """
    shapes = np.array([tensor.shape for tensor in tensor_list])
    target_shape = shapes.max(axis=0)
    padded_tensors = []
    for tensor in tensor_list:
        current_shape = np.array(tensor.shape)
        if (current_shape == target_shape).all():
            padded_tensors.append(tensor)
        else:
            # Initialize symmetric padding dimensions
            pad_dims = [(0, 0) for _ in range(len(target_shape))] 
            for dim in range(len(target_shape)):
                if current_shape[dim] < target_shape[dim]:
                    pad_total = target_shape[dim] - current_shape[dim]
                    pad_before = pad_total // 2
                    pad_after = pad_total - pad_before
                    pad_dims[dim] = (pad_before, pad_after)
            padded_tensor = Pad(to_pad=pad_dims)(tensor)
            padded_tensors.append(padded_tensor)
    return padded_tensors

class LoadingDataset(Dataset):
    def __init__(self, data_paths, ah_normalizer, transform=None, augment=None, train=False):
        self.data_paths = data_paths
        self.ah_normalizer = ah_normalizer
        self.transform = transform
        self.augment = augment
        self.train = train

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        data = torch.load(data_path)
        volume = data["images"]
        if self.transform is not None:
            volume = self.transform(volume)
        if self.train and self.augment is not None:
            volume = self.augment(volume)
        aortic_hu = data["aortic_hu"]
        aortic_hu = np.array(aortic_hu, dtype=np.float16)
        aortic_hu = aortic_hu.reshape(-1, 1)
        aortic_hu = self.ah_normalizer.transform(aortic_hu)
        # aortic_hu = aortic_hu.reshape(1, -1)
        aortic_hu = torch.tensor(aortic_hu)
        label = np.array(data["label"], dtype=np.float16)
        return volume, aortic_hu, label

class InferenceDataset(Dataset):
    def __init__(self, image_files, aortic_hues ,transform, ah_normalizer):
        self.image_files = image_files
        self.aortic_hues = aortic_hues
        self.transform = transform
        self.ah_normalizer = ah_normalizer

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_paths = self.image_files[index]
        img_list = [self.transform(path) for path in img_paths]
        aortic_hu = self.aortic_hues[index]
        aortic_hu = np.array(aortic_hu, dtype=np.float16)
        aortic_hu = aortic_hu.reshape(-1, 1)
        if len(img_list) > 1:
            aortic_hu = self.ah_normalizer.transform(aortic_hu)
            aortic_hu = torch.tensor(aortic_hu)
            padded_img_list = max_shape_pad(img_list)
            img = torch.cat(padded_img_list, 0)
            return img, aortic_hu
        else:
            aortic_hu = self.ah_normalizer.transform(aortic_hu)
            aortic_hu = torch.tensor(aortic_hu)
            return img_list[0], aortic_hu

class Ensure4D(Transform):
    """Ensure the image is 4D by adding a channel dimension and padding according to
    the kernel_size of the 3D CNN.

    Args:
        kernel_size (int): kernel size of the 3D CNN.

    Returns:
        torch.Tensor: 4D tensor.

    NOTE: This is helpful for the inference on the non-hidden dataset, as it contains 2D images.
    """

    def __init__(self, kernel_size=40):
        self.padder = Pad(to_pad=[(0,0), (0,0), (0,0), (kernel_size//2, kernel_size//2)])

    def __call__(self, img):
        if len(img.shape)<4:
            img = img.unsqueeze(-1)
            img = self.padder(img) 
            return img
        else:
            return img
        
def group_columns_by_prefix(df, grouping_list):
    """
    Groups DataFrame columns based on a list of strings.

    Args:
        df (pd.DataFrame): The DataFrame to be grouped.
        grouping_list (list): A list of strings to use as prefixes for grouping.

    Returns:
        dict: A dictionary where keys are the items in grouping_list, and values
              are DataFrames containing columns matching each prefix.
    """
    grouped_dfs = {}

    for item in grouping_list:
        filtered_cols = [col for col in df.columns if col.startswith(item)]
        grouped_df = df[filtered_cols]
        grouped_dfs[item] = grouped_df

    return grouped_dfs

def normalize_grouped_dataframes(grouped_dfs):
    """
    Normalize each row of the grouped DataFrames by their L1 norm and
    convert them back to the ungrouped DataFrame format.

    Args:
        grouped_dfs (dict): A dictionary of grouped DataFrames.

    Returns:
        pd.DataFrame: A DataFrame containing the normalized data with the original column names.
    """
    # Create an empty DataFrame with the same index as one of the grouped DataFrames
    result_df = pd.DataFrame(index=next(iter(grouped_dfs.values())).index)

    for item, grouped_df in grouped_dfs.items():
        # Normalize each row by L1 norm
        normalized_rows = grouped_df.div(grouped_df.sum(axis=1), axis=0)
        
        # Merge the normalized rows into the result DataFrame
        result_df = pd.concat([result_df, normalized_rows], axis=1)

    return result_df

class ResilientLoadImage(LoadImage):
    """ Custom LoadImage class to handle errors when loading images with SimpleITK. This reader is slower
    than the default reader, but it handles errors better.

    NOTE: This is specially helpful for inference on the hidden dataset.
    """
    def __call__(self, filename, reader=None):
        try:
            output = super().__call__(filename, reader)
        except:
            itk_loadimage = LoadImage(reader="ITKReader", image_only=True, ensure_channel_first=False, dtype=torch.float16)
            output = itk_loadimage(filename)
        return output