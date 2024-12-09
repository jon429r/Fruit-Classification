from simple_cnn import SimpleCNN,main,update_best_config
from nets import AlexNet,ResNet50,ResNet101,ResNet152
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
import torch

NUM_CLASSES=5

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
        '-p', '--model_path',
        type=str,
        required=True,
        help='The models class'
)

# Add the model path argument
parser.add_argument(
        '-m', '--model_name',
        type=str,
        required=True,
        help='Path to the model weights file'
)


# Parse arguments
args = parser.parse_args()


model_class=globals()[args.model_name]
model=model_class(NUM_CLASSES)
model.load_state_dict(torch.load(args.model_path,weights_only=True))

# Define labels
labels=["Apple","Banana","Grape","Mango","Strawberry"]
with torch.no_grad():
    outputs = F.sigmoid(model(load_and_preprocess_image(args.image_path)))
    print(f"The model predicted that the image is an {labels[torch.argmax(outputs).item()]}")
