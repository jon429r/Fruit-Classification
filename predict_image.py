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
            transforms.Resize(
                size=(128, 128),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # Apply preprocessing
    image_tensor = preprocess(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)  # Shape: [1, C, H, W]
    
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
model=SimpleCNN()
model.load_state_dict(torch.load(args.model_path,weights_only=True))

#
with torch.no_grad():
    print(model(load_and_preprocess_image(args.image_path)))

