import torch
import json
import os
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
from dataloader import get_data_loaders
import torchvision.transforms as transforms
import torch.nn.functional as F
import tabulate
import matplotlib.pyplot as plt
from Parser import FruitClassificationParser



def load_model(weights_path):
    """
    Loads the pre-trained model from the given weights file.
    """
    model = SimpleCNN()
    model.load_state_dict(torch.load(weights_path))
    model.eval()  # Set the model to evaluation mode
    return model

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pooled = self.avg_pool(x).view(x.size(0), -1)
        max_pooled = self.max_pool(x).view(x.size(0), -1)
        avg_out = self.fc(avg_pooled)
        max_out = self.fc(max_pooled)
        out = avg_out + max_out
        return self.sigmoid(out).view(x.size(0), x.size(1), 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(
            2, 1, kernel_size, padding=(kernel_size // 2), bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(  # Additional convolutional block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(  # Another convolutional block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.cam = ChannelAttention(128)  # Channel attention for 128 channels
        self.sam = SpatialAttention()  # Spatial attention

        # Adjust the fully connected layer size for the new output size after the additional layers
        self.fc1 = nn.Linear(
            128 * 8 * 8, 512
        )  # 128 channels, 8x8 spatial size after pooling
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (N, 16, 64, 64)
        x = self.pool(F.relu(self.conv2(x)))  # (N, 32, 32, 32)
        x = self.pool(F.relu(self.conv3(x)))  # (N, 64, 16, 16)
        x = self.pool(F.relu(self.conv4(x)))  # (N, 128, 8, 8)

        # Apply Channel Attention
        cam_out = self.cam(x)
        x = x * cam_out  # Element-wise multiplication

        # Apply Spatial Attention
        sam_out = self.sam(x)
        x = x * sam_out  # Element-wise multiplication

        # Flatten and classify
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(train_loader, valid_loader, model, criterion, optimizer, num_epochs, graph, device):
    epoch_losses = []
    epoch_accuracies = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update running loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Run validation for each epoch 
        model.eval()
        validation_acc = validate_model(valid_loader,model,device)

            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_acc)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Acc. : {epoch_acc:.2f}%, Val Acc. : {validation_acc:.2f}%"
        )

    # Plot learning curves
    if graph:
        plot_learning_curves(epoch_losses, epoch_accuracies, num_epochs)

def plot_learning_curves(losses, accuracies, num_epochs):
    epochs = range(1, num_epochs + 1)

    # Plot Loss curve
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label="Training Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.grid(True)

    # Plot Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label="Training Accuracy", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy over Epochs")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def test_model(test_loader, model, device):
    total = 0
    correct = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    return test_acc


def validate_model(valid_loader, model, device):
    total = 0
    correct = 0
    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    validation_acc = 100 * correct / total
    return validation_acc


def main(args,model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Apply data transformations

    transform = transforms.Compose(
        [
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    )
                ],
                p=0.1,
            ),
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(
                size=(128, 128), scale=(0.8, 1.0), ratio=(0.75, 1.33)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_loader, valid_loader, test_loader = get_data_loaders(
        args.batch_size, transform
    )

    if args.train:
        if not args.save:
            print(
                "[Warning] You are training the model but not saving it. Use `--save` to save the model."
            )

        # Train the model
        train_model(train_loader, valid_loader, model, criterion, optimizer, args.epochs, args.graph, device)
    else:
        # Attempt to load a pre-trained model
        try:
            model.load_state_dict(torch.load(f"{args.model_name}.pth"))
            print("Pre-trained model loaded successfully.")
        except FileNotFoundError:
            print(
                "[Error] Pre-trained model not found. Switching to training mode as `--train` is required."
            )
            train_model(train_loader, model, criterion, optimizer, args.epochs, device)

    # Save the model if `--save` is set
    if args.save:
        torch.save(model.state_dict(), f"{args.model_name}.pth")
        print(f"Model saved as '{args.model_name}.pth'.")

    test_acc = test_model(test_loader, model, device)

    val_acc = validate_model(valid_loader, model, device)

    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")

    return (test_acc, val_acc)


def load_existing_results(filename="results.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return {}


# Function to update the best configuration
def update_best_config(results, filename="results.json"):
    existing_results = load_existing_results(filename)

    best_existing_val_acc = existing_results.get("Validation Accuracy", 0)

    if results["Validation Accuracy"] > best_existing_val_acc:
        updated_results = {
            "Best Configuration": results,
            **existing_results,
        }
    else:
        updated_results = existing_results
        updated_results["Most Recent Configuration"] = results

    with open(filename, "w") as f:
        json.dump(updated_results, f, indent=4)

