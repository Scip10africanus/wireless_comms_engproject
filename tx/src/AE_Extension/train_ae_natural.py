"""
Advanced Training Script for TwoLayerAE using Natural Images (CIFAR-10)
Optimized for Generalization and Robustness.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import os

from AE_training import TwoLayerAE


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def main():
    device = get_device()
    print(f"🚀 Using compute device: {device.type.upper()}")

    # ---------------------------------------------------------
    # Core modification: use real natural images + strong data augmentation
    # ---------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((64, 64)),                 # Force resize to 64x64 as required by the system
        transforms.Grayscale(num_output_channels=1), # CIFAR10 is RGB; convert to single-channel grayscale

        # Data augmentation: key to improving generalization ability
        transforms.RandomHorizontalFlip(p=0.5),      # 50% probability of horizontal flip
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2)  # Randomly change brightness and contrast
        ], p=0.5),

        transforms.ToTensor(),
    ])

    print("📦 Downloading and loading CIFAR-10 natural image dataset...")
    try:
        # Use CIFAR-10 instead of FashionMNIST
        train_set = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )
        # Increase batch size to speed up training and make gradient descent smoother
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=128,
            shuffle=True
        )
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    # Initialize model and optimizer
    model = TwoLayerAE().to(device)

    # Slightly lower learning rate (1e-3 -> 5e-4) since natural image features
    # are more complex and need smoother convergence
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()

    # Since CIFAR-10 is larger and more complex, we recommend training for 30 epochs
    epochs = 30
    print(f"🔥 Starting natural-image generalization training, total epochs: {epochs}")

    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]", leave=True)
        total_loss = 0.0

        for images, _ in loop:
            images = images.to(device)

            outputs = model(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch + 1} finished | Average loss: {avg_loss:.4f}")

    # Overwrite and save weight file
    save_path = "two_layer_ae.pth"
    torch.save(model.state_dict(), save_path)
    print(f"🎉 Generalized model training finished! Weights saved to: {os.path.abspath(save_path)}")
    print("👉 Now you can directly run `benchmark_codecs_snr.m` in MATLAB to test !")

if __name__ == "__main__":
    main()