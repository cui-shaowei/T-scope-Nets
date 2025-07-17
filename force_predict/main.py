# force_predict/main.py

"""
Main inference script.
Loads the trained model, iterates over the dataset, and prints predictions together with ground-truth forces.
"""

from utils.CustomDataset import MyDataset
from utils.model import CustomResNet18
import torch
import torchvision.transforms as T
import os

# Resolve base directory relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
BINARY_DIR = os.path.join(DATA_DIR, "binarydata")
CSV_PATH = os.path.join(DATA_DIR, "data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pth")

# Select device automatically
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard ImageNet preprocessing
transform = T.Compose([
    
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

# Build dataset and dataloader
dataset = MyDataset(root_dir=BINARY_DIR, csv_path=CSV_PATH, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

# Instantiate and load model
model = CustomResNet18(num_classes=3).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Inference loop
with torch.no_grad():
    for imgs, true_forces, paths in loader:
        imgs = imgs.to(device)
        preds = model(imgs).cpu()
        for pred, true, path in zip(preds, true_forces, paths):
            pred = pred.numpy()
            true = true.numpy()
            print(f"{os.path.basename(path)}: "
                  f"pred=[{pred[0]:.4f}, {pred[1]:.4f}, {pred[2]:.4f}], "
                  f"true=[{true[0]:.4f}, {true[1]:.4f}, {true[2]:.4f}]")