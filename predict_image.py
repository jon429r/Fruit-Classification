from simple_cnn import SimpleCNN
from PIL import Image
import torchvision.transforms as transforms
import argparse
import torch

def load_and_preprocess_image(image_path):
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Define preprocessing pipeline
    preprocess = transforms.Compose(
        [
            transforms.Resize(size=(128, 128)),  # Ensure the image is resized to 128x128
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # Apply preprocessing
    image_tensor = preprocess(image)
    
    # Add batch dimension (Shape: [1, C, H, W])
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

parser = argparse.ArgumentParser(
    description='Process images with specified parameters',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Add the image path argument
parser.add_argument(
    '-i', '--image_path',
    type=str,
    required=True,
    help='Path to the input image file'
)

# Add the model path argument
parser.add_argument(
    '-m', '--model_path',
    type=str,
    required=True,
    help='Path to the model weights file'
)

# Parse arguments
args = parser.parse_args()

# Load the trained model
model = SimpleCNN(num_classes=5)  # Ensure you specify the number of classes
model.load_state_dict(torch.load(args.model_path))  # Load the model weights

# Set the model to evaluation mode
model.eval()

# Preprocess the image and make predictions
image_tensor = load_and_preprocess_image(args.image_path)

# Send the image tensor to the same device as the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
image_tensor = image_tensor.to(device)

# Make prediction without tracking gradients
with torch.no_grad():
    output = model(image_tensor)
    _, predicted_class = torch.max(output, 1)  # Get the class with the highest score

# Print the predicted class
print(f"Predicted class: {predicted_class.item()}")

