"""
Optimized Training Script for TwoLayerAE using FashionMNIST.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import os

# Ensure importing the AE model you defined
from AE_training import TwoLayerAE

def get_device():
    """Smartly detect the best compute device (NVIDIA GPU / Apple Silicon / CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Optimized for Mac M-series chips
    else:
        return torch.device("cpu")

def main():
    # 1. Smart device selection
    device = get_device()
    print(f"Using compute device: {device.type.upper()}")

    # 2. Image preprocessing (FashionMNIST is 28x28 by default; resize to 64x64)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    # 3. Load FashionMNIST dataset (fast, stable, minimal errors)
    print("Loading FashionMNIST dataset...")
    try:
        train_set = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        return

    # 4. Initialize model, optimizer, and loss function
    model = TwoLayerAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 5. Start training
    epochs = 10
    print(f"Starting training. Total epochs: {epochs}")
    
    for epoch in range(epochs):
        model.train()
        # Wrap tqdm progress bar
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=True)
        total_loss = 0
        
        for images, _ in loop:
            # Move data to the selected device
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, images)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar metrics
            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        # Print average loss for the current epoch
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

    # 6. Save weights
    save_path = "two_layer_ae.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    main()