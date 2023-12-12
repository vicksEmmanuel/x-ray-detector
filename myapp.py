import torch.nn.functional as F
from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import torch.nn as nn
import io



 # Load your model
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3
model = models.densenet121(pretrained=False)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, NUM_CLASSES)
model.load_state_dict(torch.load("model.pth"))
print("Model loaded")
# model.to(DEVICE)
model.eval()



def create_app():
    app = Flask(__name__)

    @app.route('/')
    def welcome():
        return "Welcome to the API!"

    @app.route('/predict', methods=['POST'])
    def predict():
        if request.method == 'POST':
            print("About to read image")
            # Convert the Image from request to a PIL Image
            image = Image.open(io.BytesIO(request.files['image'].read()))

            print("About to process image")
            # Preprocess the image and prepare it for your model
            preprocess =  transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0)

            print("About to run inference")

            # Run inference
            with torch.no_grad():
                output = model(input_batch)
                probabilities = F.softmax(output, dim=1)
                top_prob, top_class = probabilities.topk(1, dim=1)

            # Get the class label and probability
            class_labels = ["covid", "normal", "pneumonia"]
            predicted_label = class_labels[top_class.item()]
            predicted_prob = top_prob.item()

            # Define a confidence threshold
            confidence_threshold = 0.7

            # Check if the image is worth classifying
            if predicted_prob < confidence_threshold:
                print("Image not suitable for classification or too ambiguous.")
                return jsonify(result={
                    "class": "unknown",
                    "confidence": 0
                })
            else:
                print(f"Predicted class: {predicted_label} with confidence {predicted_prob}")
                return jsonify(result={
                    "class": predicted_label,
                    "confidence": predicted_prob
                })

    return app
