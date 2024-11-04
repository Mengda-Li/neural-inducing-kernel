import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.models import ResNet50_Weights


random_crop_flip = v2.Compose(
    [
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.PILToTensor(),
    ]
)


class FiveKDataset(Dataset):
    def __init__(self, original_dir, target_dir, transform=None):
        self.original_dir = original_dir
        self.target_dir = target_dir
        self.transform = transform
        self.file_names = [
            f
            for f in os.listdir(original_dir)
            if f.startswith("a") and f.endswith(".png")
        ]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # Construct the file path for the input and target images
        original_path = os.path.join(self.original_dir, self.file_names[idx])
        target_path = os.path.join(self.target_dir, self.file_names[idx])

        # Load the images
        original_image = Image.open(original_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        original_image, target_image = random_crop_flip([original_image, target_image])

        # Apply transformations if any
        if self.transform:
            transformed_original_image = self.transform(original_image)
            # target_image = self.transform(target_image)

        original_image, target_image = original_image.half(), target_image.half()
        return original_image, target_image, transformed_original_image


if __name__ == "__main__":
    # Define the directories
    original_dir = "data/Original"
    target_dir = "data/expertC"

    # Define any transformations, if needed
    resnet_transform = ResNet50_Weights.DEFAULT.transforms()
    # Create dataset and dataloader
    dataset = FiveKDataset(
        original_dir=original_dir, target_dir=target_dir, transform=resnet_transform
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Example usage
    for i, (original, target, transformed) in enumerate(dataloader):
        print(f"Batch {i + 1}:")
        print("Original images shape:", original.shape)
        print("Target images shape:", target.shape)

        break  # To show just the first batch
