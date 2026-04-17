"""
Evaluate and Visualize TwoLayerAE on FashionMNIST.
"""
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# 导入你的模型
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
    print(f"🔍 正在使用计算设备: {device.type.upper()}")

    # 1. 保持与训练时完全一致的预处理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    # 2. 加载测试集 (train=False 表示这是模型从未见过的全新数据)
    print("📦 正在加载测试数据...")
    test_set = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    # 取出一个批次 (比如 8 张图) 用来展示
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=True)
    
    # 3. 初始化模型并加载你刚训练好的权重
    model = TwoLayerAE().to(device)
    weight_path = "two_layer_ae.pth"
    try:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"✅ 成功加载模型权重: {weight_path}")
    except FileNotFoundError:
        print(f"❌ 找不到权重文件 {weight_path}，请确保你已经跑完了训练脚本！")
        return

    # 设置为评估模式 (这会关闭 Dropout 和 BN 等训练时特有的操作)
    model.eval()

    # 4. 抽取图像并进行推理
    dataiter = iter(test_loader)
    original_images, _ = next(dataiter)
    original_images = original_images.to(device)

    # 开启无梯度模式，加速推理
    with torch.no_grad():
        reconstructed_images = model(original_images)

    # 5. 使用 Matplotlib 画图对比
    print("🎨 正在生成对比图...")
    # 把张量转移回 CPU 并转换为 Numpy 数组方便画图
    original_images = original_images.cpu()
    reconstructed_images = reconstructed_images.cpu()

    num_images = 8
    fig, axes = plt.subplots(2, num_images, figsize=(15, 4))
    fig.suptitle("Two-Layer Autoencoder Compression Results (FashionMNIST)", fontsize=16, fontweight='bold')

    for i in range(num_images):
        # 第一行：显示原图
        ax = axes[0, i]
        # 图片形状是 [1, 64, 64]，去掉通道维度变成 [64, 64]
        ax.imshow(original_images[i].squeeze(), cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title("Original (Tx)", fontsize=12, fontweight='bold')

        # 第二行：显示重建图 (解压后)
        ax = axes[1, i]
        ax.imshow(reconstructed_images[i].squeeze(), cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title("Reconstructed (Rx)", fontsize=12, fontweight='bold')

    plt.tight_layout()
    # 弹出图像窗口
    plt.show()

if __name__ == "__main__":
    main()