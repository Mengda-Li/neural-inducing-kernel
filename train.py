import torch
import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision.models import ResNet50_Weights
from data import FiveKDataset
from module import Resnet_kernel_w
from loss import DeltaE_CIEDE2000_Loss

original_dir = "data/Original"
target_dir = "data/expertC"

# Define any transformations, if needed
resnet_transform = ResNet50_Weights.DEFAULT.transforms()
# Create dataset and dataloader
dataset = FiveKDataset(
    original_dir=original_dir, target_dir=target_dir, transform=resnet_transform
)
generator1 = torch.Generator().manual_seed(42)
train_dataset, valid_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1], generator=generator1)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
# valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True, num_workers=7)

# model = Resnet_kernel_w(loss=DeltaE_CIEDE2000_Loss)
model = Resnet_kernel_w(frozen_resnet=False)
# train with both splits
trainer = L.Trainer(accelerator="mps", devices=1)
trainer.fit(model, train_loader, valid_loader)

