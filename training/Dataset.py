import os
import io
import tarfile as tf
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset

class SpacecraftImagesDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        """
        Object for handling the spacecraft images dataset
        https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

        Args:
            annotations_file (str): Path to data labels
            img_dir (str): Path to images
            transform (lambda): Transform to be applied to images
        """
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Get the image name and label associated with the given idx
        img_name = self.img_labels.iloc[idx, 0]
        label = self.img_labels.iloc[idx, 1]

        # Get which tar file to grab image from
        tar_path = os.path.join(self.img_dir, img_name[0] + ".tar")
        
        # Grab image from tar file
        with tf.open(tar_path, "r") as src_tar:
            member = src_tar.getmember("images/" + img_name + ".png")
            image = src_tar.extractfile(member)
            image = Image.open(io.BytesIO(image.read()))

        # Apply transform if input
        if self.transform:
            image = self.transform(image)

        return image, label