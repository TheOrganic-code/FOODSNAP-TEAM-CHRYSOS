import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
import json
import os

# =========================================================
# 1️⃣ Parse command-line arguments
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True, help="Path to the image file")
args = parser.parse_args()

# =========================================================
# 2️⃣ Paths and setup
# =========================================================
MODEL_PATH = "backend/food_model.pth"
CLASS_INDEX_PATH = "backend/class_indices.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================================================
# 3️⃣ Load model architecture
# =========================================================
from torchvision.models import efficientnet_b0

class FoodClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FoodClassifier, self).__init__()
        self.base_model = efficientnet_b0(weights=None)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# =========================================================
# 4️⃣ Load checkpoint
# =========================================================
checkpoint = torch.load(MODEL_PATH, map_location=device)
num_classes = checkpoint.get("num_classes", 80)
model = FoodClassifier(num_classes)

# ✅ Load weights safely (ignore mismatched layers)
state_dict = checkpoint["model_state_dict"]
model_dict = model.state_dict()
pretrained_dict = {
    k: v for k, v in state_dict.items()
    if k in model_dict and v.shape == model_dict[k].shape
}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

model = model.to(device)
model.eval()

# =========================================================
# 5️⃣ Food name mapping (Indian dishes)
# =========================================================
idx_to_class = {
    0: "Aloo Gobi", 1: "Aloo Matar", 2: "Aloo Paratha", 3: "Aloo Tikki",
    4: "Baingan Bharta", 5: "Bhindi Masala", 6: "Biryani", 7: "Butter Chicken",
    8: "Chaat", 9: "Chana Masala", 10: "Chapati", 11: "Chicken Curry",
    12: "Chicken Tikka", 13: "Chole Bhature", 14: "Daal", 15: "Dal Makhani",
    16: "Dhokla", 17: "Dosa", 18: "Fish Curry", 19: "Fried Rice",
    20: "Gajar Halwa", 21: "Gulab Jamun", 22: "Halwa", 23: "Idli",
    24: "Jalebi", 25: "Jeera Rice", 26: "Kadai Paneer", 27: "Kheer",
    28: "Khichdi", 29: "Lassi", 30: "Matar Paneer", 31: "Momos",
    32: "Naan", 33: "Pani Puri", 34: "Paneer Butter Masala", 35: "Paneer Tikka",
    36: "Papad", 37: "Pav Bhaji", 38: "Poha", 39: "Pongal", 40: "Pulao",
    41: "Raita", 42: "Rajma Chawal", 43: "Rasgulla", 44: "Roti",
    45: "Sabudana Khichdi", 46: "Sambar", 47: "Samosa", 48: "Sandwich",
    49: "Sarson da Saag", 50: "Sev Puri", 51: "Shahi Paneer",
    52: "Tandoori Chicken", 53: "Thali", 54: "Upma", 55: "Vada Pav",
    56: "Vegetable Curry", 57: "Veg Fried Rice", 58: "Veg Pulao",
    59: "Veg Thali", 60: "Mysore Pak", 61: "Pesarattu", 62: "Rava Dosa",
    63: "Onion Pakoda", 64: "Pakora", 65: "Poori", 66: "Masala Dosa",
    67: "Gobi Manchurian", 68: "Noodles", 69: "Chilli Paneer",
    70: "Chilli Chicken", 71: "Egg Curry", 72: "Chicken Biryani",
    73: "Mutton Curry", 74: "Paneer Roll", 75: "Egg Roll", 76: "Dal Tadka",
    77: "Palak Paneer", 78: "Rasam", 79: "Curd Rice"
}

# =========================================================
# 6️⃣ Nutrition facts (simplified database)
# =========================================================
nutrition_data = {
    "Samosa": {"calories": 262, "carbs": 32, "protein": 4, "fats": 14, "tip": "Deep-fried; high in fats — enjoy occasionally."},
    "Paneer Butter Masala": {"calories": 410, "carbs": 14, "protein": 12, "fats": 35, "tip": "Rich in fats and protein; limit butter intake."},
    "Biryani": {"calories": 320, "carbs": 42, "protein": 9, "fats": 12, "tip": "Balanced but can be high in oil — use brown rice for healthier version."},
    "Chole Bhature": {"calories": 450, "carbs": 48, "protein": 10, "fats": 20, "tip": "Very high calorie; best for occasional treat."},
    "Idli": {"calories": 58, "carbs": 12, "protein": 2, "fats": 0.4, "tip": "Light and healthy breakfast option."},
    "Dosa": {"calories": 133, "carbs": 16, "protein": 3, "fats": 6, "tip": "Crispy and light — pair with sambar for protein."},
    "Pav Bhaji": {"calories": 400, "carbs": 50, "protein": 8, "fats": 18, "tip": "High in carbs and butter; enjoy moderately."},
    "Poha": {"calories": 180, "carbs": 32, "protein": 3, "fats": 4, "tip": "Good breakfast; low fat if cooked with less oil."},
    "Rajma Chawal": {"calories": 270, "carbs": 45, "protein": 9, "fats": 5, "tip": "High protein vegetarian option."},
    "Dal Makhani": {"calories": 300, "carbs": 26, "protein": 12, "fats": 16, "tip": "Protein-rich but creamy — eat in moderation."},
    "Gulab Jamun": {"calories": 150, "carbs": 30, "protein": 2, "fats": 6, "tip": "High in sugar — small portions recommended."}
}

# =========================================================
# 7️⃣ Load and preprocess the image
# =========================================================
if not os.path.exists(args.image):
    raise FileNotFoundError(f"❌ Image not found: {args.image}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img = Image.open(args.image).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

# =========================================================
# 8️⃣ Predict
# =========================================================
with torch.no_grad():
    outputs = model(img_tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    conf, predicted = torch.max(probs, 1)
    confidence = conf.item() * 100
    pred_idx = predicted.item()

predicted_food = idx_to_class.get(pred_idx, f"class_{pred_idx}")

# =========================================================
# 9️⃣ Nutrition Info Display
# =========================================================
if predicted_food in nutrition_data:
    info = nutrition_data[predicted_food]
    print(f"\n🍽️ Predicted Food: {predicted_food}")
    print(f"🤖 Confidence: {confidence:.2f}% sure")
    print(f"🔥 Calories: {info['calories']} kcal")
    print(f"🥔 Carbs: {info['carbs']}g | 🍗 Protein: {info['protein']}g | 🧈 Fats: {info['fats']}g")
    print(f"❤️ Health Tip: {info['tip']}")
else:
    print(f"\n🍽️ Predicted Food: {predicted_food}")
    print(f"🤖 Confidence: {confidence:.2f}% sure")
    print("⚠️ Nutrition data not available for this item.")
