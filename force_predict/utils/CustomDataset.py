# utils/CustomDataset.py

"""
Dataset wrapper for force-vector regression.
Reads images from a directory and loads corresponding (x, y, z) labels from a CSV file.
Expected CSV format:
image_path,x,y,z
0001.jpg,0.1,0.2,-0.3
...
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):

    """
    Args:
        root_dir (str): Directory containing images.
        csv_path (str): CSV file with columns image_path,x,y,z.
        transform (callable, optional): Transform applied to each image.
    """

    def __init__(self, root_dir, csv_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Load CSV and build mapping: image_name -> force tensor
        df = pd.read_csv(csv_path)
        self.label_dict = {}

        for _, row in df.iterrows():
            name = os.path.basename(row['image_path'])
            self.label_dict[name] = torch.tensor([row['x'], row['y'], row['z']], dtype=torch.float32)

        # List all valid image files that have labels
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        self.img_paths = [
            os.path.join(root_dir, f)
            for f in sorted(os.listdir(root_dir))
            if f.lower().endswith(valid_ext) and f in self.label_dict
        ]

        if len(self.img_paths) == 0:
            raise RuntimeError(f"No labeled images found in {root_dir}")

    def __len__(self):
        
        """Return the number of samples."""
        
        return len(self.img_paths)

    def __getitem__(self, idx):
        
        """
        Args:
            idx (int): Sample index.
        Returns:
            img (Tensor): Transformed image.
            true_force (Tensor): Ground-truth force vector (x, y, z).
            img_path (str): Full path to the image file.
        """
        
        img_path = self.img_paths[idx]
        img_name = os.path.basename(img_path)
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        true_force = self.label_dict[img_name]
        
        return img, true_force, img_path