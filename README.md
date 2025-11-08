# FOODSNAP-TEAM-CHRYSOS
🍽️ AI model that recognizes Indian dishes from photos and tells you what’s on your plate including calories, nutrients, and a quick health tip!
# 🍛 Indian Food Image Classifier with Nutrition Insights

A deep learning project that identifies **80+ Indian dishes** from images and provides their **nutritional breakdown** — built with **PyTorch** and **EfficientNet-B0**.

---

## 📸 Overview

This AI model classifies popular Indian foods and gives their nutrition facts like **calories**, **carbs**, **protein**, **fats**, and a short **health tip**.

The model was trained using the [Indian Food Images Dataset](https://www.kaggle.com/datasets/swapnilbhange/indian-food-images) and runs seamlessly on both **CPU** and **GPU**.

---

## ✨ Features

- 🍽️ Classifies 80+ Indian dishes  
- 🔥 Displays nutrition data (calories, carbs, protein, fats)  
- ❤️ Gives a health tip for each food  
- ⚙️ Built with PyTorch + EfficientNet-B0  
- 💻 Works on both CPU and GPU  

---

## 🧠 Tech Stack

| Component | Technology |
|------------|-------------|
| **Language** | Python |
| **Framework** | PyTorch |
| **Model** | EfficientNet-B0 |
| **Dataset** | [Kaggle - Indian Food Images](https://www.kaggle.com/datasets/swapnilbhange/indian-food-images) |
| **Libraries** | torchvision, PIL, argparse, json |

---

## 📂 Project Structure

meow/
│
├── backend/
│ ├── train_model.py # Model training script
│ ├── predict_image.py # Prediction + Nutrition info
│ ├── class_indices.json # Class labels
│ ├── food_model.pth # Trained model weights
│ └── test_images/ # Folder for test images
│
└── README.md


📊 Example Classes
Category	Example Dishes
Breakfast	Idli, Dosa, Poha, Upma
Lunch	Dal Makhani, Rajma Chawal, Biryani
Snacks	Samosa, Pakora, Pav Bhaji
Sweets	Gulab Jamun, Jalebi, Kheer
🩸 Example Nutrition Data
Dish	Calories	Carbs	Protein	Fat	Health Tip
Samosa	262 kcal	32g	4g	14g	Deep-fried; enjoy occasionally
Idli	58 kcal	12g	2g	0.4g	Light and healthy breakfast
Biryani	320 kcal	42g	9g	12g	Try with brown rice for better nutrition
Paneer Butter Masala	410 kcal	14g	12g	35g	High in fats; limit butter intake
🧪 Future Improvements

 Add top-3 prediction display

 Add Glycemic Index & Veg/Non-Veg tag

 Deploy using Streamlit/Flask

 Add webcam food detection

 👨‍💻 Author
 TEAM CHRYSOS
