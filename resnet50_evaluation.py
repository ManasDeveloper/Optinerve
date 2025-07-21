import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from dataset import EyeDataset           
from evaluate import evaluate        

# ------------------ Configuration ------------------
img_dir = "dataset/right_eye_dataset"
annotations = "annotations.csv"
model_path = "eye_disease_resnet50.pth"
batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Transforms ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

# ------------------ Dataset & Dataloader ------------------
dataset = EyeDataset(img_dir=img_dir, annotations=annotations, transform=transform)

# Use full dataset or just test split if needed
test_size = int(0.2 * len(dataset))
_, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - test_size, test_size])

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ------------------ Model Definition ------------------
model = models.resnet50(pretrained=False)

# Replace final classifier
# replace the final classifier 
num_classes = len(dataset.label_encoder.classes_)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features,num_classes)

model.to(device)

# Load saved weights
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)

# ------------------ Evaluation ------------------
criterion = nn.CrossEntropyLoss()
evaluate(model, test_loader, criterion, device=device, label_encoder=dataset.label_encoder)
