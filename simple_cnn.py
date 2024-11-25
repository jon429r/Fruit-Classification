import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
from dataloader import get_data_loaders
import torchvision.transforms as transforms
import torch.nn.functional as F
import tabulate
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
    description="Fruit Image Classification with Simple CNN"
)

parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size for training"
)

parser.add_argument("--train", action="store_true", help="Enable training mode")
parser.add_argument("--save", action="store_true", help="Save the model")

args = parser.parse_args()


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
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(train_loader, model, criterion, optimizer, num_epochs, device):
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

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_acc)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%"
        )

    # Plot learning curves
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


def main(args):
    model = SimpleCNN(num_classes=5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Apply data transformations
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
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
        train_model(train_loader, model, criterion, optimizer, args.epochs, device)
    else:
        # Attempt to load a pre-trained model
        try:
            model.load_state_dict(torch.load("simple_cnn.pth"))
            print("Pre-trained model loaded successfully.")
        except FileNotFoundError:
            print(
                "[Error] Pre-trained model not found. Switching to training mode as `--train` is required."
            )
            train_model(train_loader, model, criterion, optimizer, args.epochs, device)

    # Save the model if `--save` is set
    if args.save:
        torch.save(model.state_dict(), "simple_cnn.pth")
        print("Model saved as 'simple_cnn.pth'.")

    test_acc = test_model(test_loader, model, device)

    val_acc = validate_model(valid_loader, model, device)

    # chart test and val
    results = tabulate.tabulate(
        [
            ["Epochs", args.epochs],
            ["Learning rate", args.lr],
            ["Batch Size", args.batch_size],
            ["Test Accuracy", test_acc],
            ["Validation Accuracy", val_acc],
        ],
        headers=["parameters", "Value"],
    )

    print(results)

    # append results to the end of the file
    with open("results.txt", "a") as f:
        f.write("\n")
        f.write(results)
        f.write("\n")


if __name__ == "__main__":
    main(args)
