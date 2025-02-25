import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the current directory to path to import from train.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train import InattentionCNN, InattentionDropout  # Import only the model classes

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# CIFAR-10 Test Data Loading
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def get_sample_images():
    """Get one sample image from each class."""
    class_images = {i: None for i in range(10)}
    class_patterns = {i: None for i in range(10)}
    
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            # Get inattention patterns
            x = model.conv1(images)
            x = F.relu(x)
            x = model.pool(x)
            x = model.conv2(x)
            
            # Extract patterns from first inattention layer
            retention_probs = []
            for layer in model.inattention1.attention_layers:
                retention_prob = layer[0](x)
                retention_probs.append(retention_prob)
            
            retention_prob = sum(retention_probs) / len(retention_probs)
            
            # Store one image per class
            for idx, label in enumerate(labels):
                label = label.item()
                if class_images[label] is None:
                    class_images[label] = images[idx].cpu()
                    class_patterns[label] = retention_prob[idx].mean(dim=0).cpu()
            
            # Break if we have all classes
            if all(v is not None for v in class_images.values()):
                break
                
    return class_images, class_patterns

def visualize_inattention_patterns(save_path='inattention_visualization.png'):
    """Create a horizontal grid visualization of inattention patterns."""
    class_images, class_patterns = get_sample_images()
    
    # Create a figure with a horizontal layout
    fig, axes = plt.subplots(2, 10, figsize=(20, 5))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # Plot original images in the first row
    for i in range(10):
        img = class_images[i].permute(1, 2, 0)
        img = img * torch.tensor([0.2470, 0.2435, 0.2616]) + torch.tensor([0.4914, 0.4822, 0.4465])
        img = torch.clamp(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'{classes[i]}', fontsize=10)
        axes[0, i].axis('off')
    
    # Plot inattention patterns in the second row
    for i in range(10):
        pattern = class_patterns[i]
        im = axes[1, i].imshow(pattern, cmap='viridis')
        axes[1, i].set_title('Inattention', fontsize=10)
        axes[1, i].axis('off')
    
    # Add a colorbar
    plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label='Inattention Strength')
    
    plt.suptitle('Inattention Patterns Across CIFAR-10 Classes', fontsize=14, y=1.05)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def visualize_stack_weights(save_path='stack_weights.png'):
    """Visualize the learned weights of different stacks in the inattention layers."""
    alpha_1 = F.softmax(model.inattention1.alpha, dim=0).detach().cpu()
    alpha_2 = F.softmax(model.inattention2.alpha, dim=0).detach().cpu()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    x = np.arange(len(alpha_1))
    width = 0.6
    
    # Plot with better styling
    ax1.bar(x, alpha_1, width, color='skyblue', edgecolor='black')
    ax1.set_title('Inattention Layer 1\nStack Weights')
    ax1.set_xlabel('Stack Index')
    ax1.set_ylabel('Weight')
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(x, alpha_2, width, color='lightcoral', edgecolor='black')
    ax2.set_title('Inattention Layer 2\nStack Weights')
    ax2.set_xlabel('Stack Index')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    # Load the trained model
    model = InattentionCNN().to(device)
    model.load_state_dict(torch.load('inattention_cnn_best.pth'))
    model.eval()
    
    # Generate visualizations
    visualize_inattention_patterns()
    visualize_stack_weights()
    print("Visualizations have been saved!")
