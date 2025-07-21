from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import transforms
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
import os
from PIL import Image

class EyeDataset(Dataset):
    def __init__(self,img_dir,annotations,transform=None):
        self.data = pd.read_csv(annotations)
        self.img_dir = img_dir
        self.transform = transform

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.data['labels'])


    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        image_path = os.path.join(self.img_dir,self.data.iloc[idx]['Right-Fundus'])
        image = Image.open(image_path).convert("RGB")

        label_str = self.data.iloc[idx]['labels']
        label_idx = self.label_encoder.transform([label_str])[0]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label_idx,dtype = torch.long)