
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import rsna_config as config

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
    
def apply_augmentation(images, labels, p=0.1):
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