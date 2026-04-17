"""
Evaluate and Visualize TwoLayerAE on FashionMNIST.
"""
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Import your model
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
    print(f"🔍 Using compute device: {device.type.upper()}")

    # 1. Keep preprocessing fully consistent with training
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    # 2. Load test set (train=False means unseen data for the model)
    print("📦 Loading test data...")
    test_set = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    # Fetch one batch (e.g., 8 images) for visualization
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=True)
    
    # 3. Initialize model and load trained weights
    model = TwoLayerAE().to(device)
    weight_path = "two_layer_ae.pth"
    try:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"✅ Successfully loaded model weights: {weight_path}")
    except FileNotFoundError:
        print(f"❌ Weight file {weight_path} not found. Please make sure training has completed.")
        return

    # Set evaluation mode (disables training-specific ops like Dropout and BN)
    model.eval()

    # 4. Fetch images and run inference
    dataiter = iter(test_loader)
    original_images, _ = next(dataiter)
    original_images = original_images.to(device)

    # Enable no-grad mode for faster inference
    with torch.no_grad():
        reconstructed_images = model(original_images)

    # 5. Visualize side-by-side comparison with Matplotlib
    print("🎨 Generating comparison plot...")
    # Move tensors back to CPU for plotting
    original_images = original_images.cpu()
    reconstructed_images = reconstructed_images.cpu()

    num_images = 8
    fig, axes = plt.subplots(2, num_images, figsize=(15, 4))
    fig.suptitle("Two-Layer Autoencoder Compression Results (FashionMNIST)", fontsize=16, fontweight='bold')

    for i in range(num_images):
        # First row: original images
        ax = axes[0, i]
        # Image shape is [1, 64, 64], squeeze channel dimension to [64, 64]
        ax.imshow(original_images[i].squeeze(), cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title("Original (Tx)", fontsize=12, fontweight='bold')

        # Second row: reconstructed images
        ax = axes[1, i]
        ax.imshow(reconstructed_images[i].squeeze(), cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title("Reconstructed (Rx)", fontsize=12, fontweight='bold')

    plt.tight_layout()
    # Show the figure window
    plt.show()

if __name__ == "__main__":
    main()
