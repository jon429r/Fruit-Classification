
import cv2
import torch
from torchvision import transforms
from PIL import Image
from simple_cnn import load_model

# Load the model
weights_path = "simple_cnn.pth"
model = load_model(weights_path)

# Define class labels
class_labels = ["Apple", "Banana", "Grape", "Mango", "Strawberry"]

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust based on your model's input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust mean and std if needed
])

# OpenCV: Initialize the webcam
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to PIL Image for processing
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Preprocess the image
    input_tensor = transform(image)

    # Run the model
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        label = class_labels[predicted]

    # Overlay prediction on the frame
    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Fruit Classifier', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
