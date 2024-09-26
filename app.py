from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Step 1: Define transformations (same as the ones used during training)
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Step 2: Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)  # Pretrained=False since we are loading our fine-tuned model
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)  # Change '2' to the number of your classes
model.load_state_dict(torch.load('./model/your_model.pth', map_location=device))  # Load the model
model = model.to(device)
model.eval()  # Set model to evaluation mode

# Step 3: Define class labels (update with your actual class names)
class_names = ['home_bell', 'home_bulb', 'home_picture_frame','home_switchboard','home_tap']  # Replace with your actual class names

# Step 4: Utility function to preprocess the image
def preprocess_image(image):
    """Preprocess the image to match the input format expected by the model."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = data_transforms(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Step 5: Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
	"""Endpoint to predict the class of an uploaded image."""
	if 'file' not in request.files:
		return jsonify({"error": "No file uploaded"}), 400

	file = request.files['file']

	try:
		# Read the image from the uploaded file
		image = Image.open(io.BytesIO(file.read()))

		# Preprocess the image
		input_tensor = preprocess_image(image)

		# Perform inference
		with torch.no_grad():
			outputs = model(input_tensor)
			_, predicted_idx = torch.max(outputs, 1)
			print("predicted_idx.item():", predicted_idx.item())
			predicted_class = class_names[predicted_idx.item()]
			print("predicted_class:", predicted_class)
		print("predicted_class1:", predicted_class)
		# Return the predicted class in a JSON response
		return jsonify({"predicted_class": predicted_class})

	except Exception as e:
		return jsonify({"error": str(e)}), 500

# Step 6: Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

