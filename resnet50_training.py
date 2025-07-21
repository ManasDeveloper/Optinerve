import torchvision.models as models 
import torch.nn as nn 
from torchvision.transforms import transforms
from torch.utils.data import random_split,WeightedRandomSampler
from torch.utils.data import DataLoader
import torch
from train import train

from dataset import EyeDataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

img_dir = "dataset/right_eye_dataset"
annotations = "annotations.csv"


# create the dataset

dataset = EyeDataset(img_dir=img_dir,annotations=annotations,transform=transform)


# split the dataset

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# creating the model
model = models.resnet50(pretrained = True)

for param in model.parameters():
    param.requires_grad = True

# replace the final classifier 
num_classes = len(dataset.label_encoder.classes_)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features,num_classes)

model.to(device)




criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)


# train the model
train(model,criterion,optimizer,10,train_loader,device)


# save the model 
torch.save(model.state_dict(),"eye_disease_resnet50.pth")
print("✅ Model saved as 'eye_disease_resnet50.pth'")




