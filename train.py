import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import os

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
batch_size = 128
num_epochs = 50
learning_rate = 0.001

# CIFAR-10 Data Loading
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Implementation of the Inattention Dropout Layer
class InattentionDropout(nn.Module):
    def __init__(self, channels, dropout_prob=0.2, num_stacks=3):
        super(InattentionDropout, self).__init__()
        self.dropout_prob = dropout_prob
        self.num_stacks = num_stacks
        self.alpha = nn.Parameter(torch.ones(num_stacks) / num_stacks)
        
        # Create stack of inattention mechanisms
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.Sigmoid()
            ) 
            for _ in range(num_stacks)
        ])
        
    def forward(self, x, training=True):
        if not training or self.dropout_prob == 0:
            return x
            
        # Normalize alphas to sum to 1
        alpha_norm = F.softmax(self.alpha, dim=0)
        
        # Initialize combined mask
        batch_size, channels, height, width = x.shape
        combined_mask = torch.ones_like(x)
        
        for k in range(self.num_stacks):
            # Generate attention-based retention probabilities
            retention_prob = self.attention_layers[k](x)
            
            # Create binary dropout mask
            mask = torch.bernoulli(retention_prob * (1 - self.dropout_prob) + self.dropout_prob)
            
            # Weight the mask by alpha and combine
            combined_mask = combined_mask * (alpha_norm[k] * mask + (1 - alpha_norm[k]))
        
        # Apply the mask and scale by inverse dropout probability
        return x * combined_mask / (1 - self.dropout_prob)

# Baseline CNN Model
class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # Changed from 4x4 to 8x8
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        batch_size = x.size(0)
        x = x.view(batch_size, 128 * 8 * 8)  # Explicitly preserve batch size
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# CNN with Inattention Dropout
class InattentionCNN(nn.Module):
    def __init__(self):
        super(InattentionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.inattention1 = InattentionDropout(64, dropout_prob=0.25, num_stacks=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.inattention2 = InattentionDropout(128, dropout_prob=0.25, num_stacks=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # Changed from 4x4 to 8x8
        self.inattention3 = nn.Dropout(0.25)  # Using standard dropout for FC layers
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.inattention1(x, self.training)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.inattention2(x, self.training)
        batch_size = x.size(0)
        x = x.view(batch_size, 128 * 8 * 8)  # Explicitly preserve batch size
        x = F.relu(self.fc1(x))
        x = self.inattention3(x)
        x = self.fc2(x)
        return x

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(train_loader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

# Evaluation function
def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc

# Function to run full training experiment
def run_experiment(model_name, model_class):
    print(f"\nRunning experiment: {model_name}")
    model = model_class().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
            # Save the best model
            torch.save(model.state_dict(), f'{model_name.lower().replace(" ", "_")}_best.pth')
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    print(f"\nBest test accuracy: {best_acc:.2f}%")
    return history

def visualize_inattention_patterns(model, test_loader, device, save_path='inattention_patterns.png'):
    """
    Visualize inattention patterns for different CIFAR-10 classes.
    """
    model.eval()
    class_images = {i: [] for i in range(10)}
    class_patterns = {i: [] for i in range(10)}
    
    # Get one batch of images
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            batch_size = images.size(0)
            
            # Store original images by class
            for idx in range(batch_size):
                label = labels[idx].item()
                if len(class_images[label]) < 1:  # Store only one image per class
                    class_images[label].append(images[idx].cpu())

            # Get inattention patterns
            x = model.conv1(images)
            x = F.relu(x)
            x = model.pool(x)
            x = model.conv2(x)
            
            # Extract patterns from first inattention layer
            retention_probs = []
            for layer in model.inattention1.attention_layers:
                retention_prob = layer[0](x)  # Get retention probabilities
                retention_probs.append(retention_prob)
            
            # Store patterns by class
            retention_prob = sum(retention_probs) / len(retention_probs)  # Average across stacks
            for idx in range(batch_size):
                label = labels[idx].item()
                if len(class_patterns[label]) < 1:  # Store only one pattern per class
                    class_patterns[label].append(retention_prob[idx].mean(dim=0).cpu())
            
            # Break if we have all classes
            if all(len(v) == 1 for v in class_images.values()):
                break
    
    # Create visualization grid
    fig, axes = plt.subplots(10, 3, figsize=(15, 40))
    plt.subplots_adjust(hspace=0.3)
    
    for class_idx in range(10):
        # Original image
        img = class_images[class_idx][0].permute(1, 2, 0)
        img = img * torch.tensor([0.2470, 0.2435, 0.2616]) + torch.tensor([0.4914, 0.4822, 0.4465])
        img = torch.clamp(img, 0, 1)
        axes[class_idx, 0].imshow(img)
        axes[class_idx, 0].set_title(f'Class: {classes[class_idx]}\nOriginal')
        axes[class_idx, 0].axis('off')
        
        # Inattention pattern
        pattern = class_patterns[class_idx][0]
        axes[class_idx, 1].imshow(pattern, cmap='viridis')
        axes[class_idx, 1].set_title('Inattention Pattern')
        axes[class_idx, 1].axis('off')
        
        # Combined visualization
        pattern_resized = F.interpolate(pattern.unsqueeze(0).unsqueeze(0), 
                                      size=(32, 32), 
                                      mode='bilinear').squeeze()
        overlay = img * pattern_resized.unsqueeze(-1).expand(-1, -1, 3)
        axes[class_idx, 2].imshow(overlay)
        axes[class_idx, 2].set_title('Combined Effect')
        axes[class_idx, 2].axis('off')
    
    plt.suptitle('Inattention Patterns Across CIFAR-10 Classes', fontsize=16, y=0.92)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def visualize_stack_weights(model, save_path='stack_weights.png'):
    """
    Visualize the learned weights of different stacks in the inattention layers.
    """
    alpha_1 = F.softmax(model.inattention1.alpha, dim=0).detach().cpu()
    alpha_2 = F.softmax(model.inattention2.alpha, dim=0).detach().cpu()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot stack weights for both inattention layers
    x = np.arange(len(alpha_1))
    width = 0.35
    
    ax1.bar(x, alpha_1, width, label='Stack Weights')
    ax1.set_title('Inattention Layer 1\nStack Weights')
    ax1.set_xlabel('Stack Index')
    ax1.set_ylabel('Weight')
    
    ax2.bar(x, alpha_2, width, label='Stack Weights')
    ax2.set_title('Inattention Layer 2\nStack Weights')
    ax2.set_xlabel('Stack Index')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Only run the training if this file is run directly
if __name__ == '__main__':
    # Run both experiments
    results = {}
    results['baseline'] = run_experiment("Baseline CNN", BaselineCNN)
    results['inattention'] = run_experiment("Inattention CNN", InattentionCNN)

    # Visualize inattention patterns and stack weights if model exists
    if os.path.exists('inattention_cnn_best.pth'):
        model = InattentionCNN().to(device)
        model.load_state_dict(torch.load('inattention_cnn_best.pth'))
        visualize_inattention_patterns(model, test_loader, device)
        visualize_stack_weights(model)

    # Plotting results
    plt.figure(figsize=(12, 5))  

    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(results['baseline']['train_acc'], label='Baseline Train')
    plt.plot(results['baseline']['test_acc'], label='Baseline Test')
    plt.plot(results['inattention']['train_acc'], label='Inattention Train')
    plt.plot(results['inattention']['test_acc'], label='Inattention Test')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(results['baseline']['train_loss'], label='Baseline Train')
    plt.plot(results['baseline']['test_loss'], label='Baseline Test')
    plt.plot(results['inattention']['train_loss'], label='Inattention Train')
    plt.plot(results['inattention']['test_loss'], label='Inattention Test')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # Save results
    plt.tight_layout()
    plt.savefig('inattention_dropout_results.png')
    plt.show()

    # Calculate and print summary statistics
    for model_name in results:
        final_train_acc = results[model_name]['train_acc'][-1]
        final_test_acc = results[model_name]['test_acc'][-1]
        best_test_acc = max(results[model_name]['test_acc'])
        
        print(f"\n{model_name.capitalize()} Summary:")
        print(f"Final Train Accuracy: {final_train_acc:.2f}%")
        print(f"Final Test Accuracy: {final_test_acc:.2f}%")
        print(f"Best Test Accuracy: {best_test_acc:.2f}%")

    # Analyze alpha values for the inattention model
    inattention_model = InattentionCNN().to(device)
    print("\nFinal alpha values for each inattention layer:")
    for i, module in enumerate(inattention_model.modules()):
        if isinstance(module, InattentionDropout):
            normalized_alpha = F.softmax(module.alpha, dim=0)
            print(f"Layer {i}: {normalized_alpha.detach().cpu().numpy()}")