import os
import numpy as np
import cv2
import shutil
import imghdr
import glob
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import requests
from io import BytesIO

# # ✅ Step 1: Define dataset path
# dataset_path = Path(r"C:\Users\rucha\Downloads\archive (1)\Acne")

# if not dataset_path.exists():
#     print("🚨 Folder not found! Check the path.")
#     exit()
# else:
#     print("✅ Folder found!")

# # ✅ Get image files
# image_files = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))
# image_files = [str(img) for img in image_files if imghdr.what(img) in ['jpeg', 'png']]
# print(f"🖼️ Found {len(image_files)} images.")

# # ✅ Load MobileNetV2 for feature extraction
# base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# def extract_features(img_path):
#     try:
#         img = image.load_img(img_path, target_size=(224, 224))
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = preprocess_input(img_array)
#         features = base_model.predict(img_array, verbose=0)
#         return features.flatten()
#     except Exception as e:
#         print(f"❌ Error processing {img_path}: {e}")
#         return None

# # ✅ Extract features for images
# feature_list = []
# valid_image_files = []
# for i, img in enumerate(image_files):
#     print(f"🔄 Processing {i+1}/{len(image_files)}: {img}")
#     features = extract_features(img)
#     if features is not None:
#         feature_list.append(features)
#         valid_image_files.append(img)

# feature_list = np.array(feature_list)

# # ✅ Apply K-Means clustering
# kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
# labels = kmeans.fit_predict(feature_list)

# # ✅ Compute silhouette score
# sil_score = silhouette_score(feature_list, labels)
# print(f"📊 Silhouette Score: {sil_score}")

# # ✅ Organize images into severity-based folders
# clustered_dataset_path = Path("clustered_dataset")
# clustered_dataset_path.mkdir(exist_ok=True)

# for i, img_path in enumerate(valid_image_files):
#     severity_label = str(labels[i] + 1)
#     target_folder = clustered_dataset_path / severity_label
#     target_folder.mkdir(exist_ok=True)
#     shutil.copy(img_path, target_folder / Path(img_path).name)

# print(f"📁 Images copied into {clustered_dataset_path}/1, 2, 3, 4 based on severity.")

# # ✅ Check if model already exists
# MODEL_PATH = "acne_severity_model.h5"
# if os.path.exists(MODEL_PATH):
#     print("✅ Pre-trained model found! Loading...")
#     model = load_model(MODEL_PATH)

#     # ✅ Recompile the model to ensure metrics are set
#     model.compile(optimizer=Adam(learning_rate=0.0001), 
#                   loss='sparse_categorical_crossentropy', 
#                   metrics=['accuracy'])
# else:
#     print("🚀 No pre-trained model found. Training a new one...")

#     # ✅ Data Preparation
#     data_gen = ImageDataGenerator(validation_split=0.2, preprocessing_function=preprocess_input)

#     train_data = data_gen.flow_from_directory(
#         clustered_dataset_path,
#         target_size=(224, 224),
#         batch_size=32,
#         class_mode='sparse',
#         subset='training'
#     )

#     val_data = data_gen.flow_from_directory(
#         clustered_dataset_path,
#         target_size=(224, 224),
#         batch_size=32,
#         class_mode='sparse',
#         subset='validation'
#     )

#     # ✅ Define CNN Model
#     model = Sequential([
#         base_model,
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dropout(0.5),
#         Dense(4, activation='softmax')
#     ])

#     model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#     # ✅ Train the Model
#     model.fit(train_data, validation_data=val_data, epochs=10)

#     # ✅ Save the Model
#     model.save(MODEL_PATH)
#     print("✅ Model saved successfully!")

# # ✅ Function to predict acne severity
# def predict_acne_severity(img_path, model_path=MODEL_PATH):
#     model = load_model(model_path)
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)
#     prediction = model.predict(img_array)
#     severity = int(np.argmax(prediction) + 1)
#     return severity

# ✅ API Integration
API_URL = "https://skintelligence.netlify.app/api/retrieve?userId=user_2smY7lfp5zgBLPvp7iUPJYKsbmS"

def fetch_image_from_api():
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        data = response.json()

        print(data)

        if "images" not in data:
            print("🚨 No images found in API response!")
            print(data)
            return None

        print(data["images"])
        return data["images"]
    
    except Exception as e:
        print(f"❌ Error fetching image from API: {e}")
        return None

# def predict_acne_severity_from_url(image_url):
#     try:
#         response = requests.get(image_url)
#         response.raise_for_status()
#         img = Image.open(BytesIO(response.content))

#         img = img.resize((224, 224))
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = preprocess_input(img_array)

#         prediction = model.predict(img_array)
#         severity = int(np.argmax(prediction) + 1)
        
#         return severity

#     except Exception as e:
#         print(f"❌ Error processing image: {e}")
#         return None

# ✅ Fetch multiple images from API and predict severity for each
image_urls = fetch_image_from_api()  # Ensure API returns a list of image URLs

if image_urls and isinstance(image_urls, list):  # Check if multiple images are returned
    severity_results = {}  # Dictionary to store results

 #   for i, img_url in enumerate(image_urls):
 #       severity = predict_acne_severity_from_url(img_url)
 #       if severity:
 #           severity_results[f"Image {i+1}"] = severity

    image_url = fetch_image_from_api()
# if image_url:
#     severity = predict_acne_severity_from_url(image_url)
#     if severity:
#         print(f"✅ Predicted Acne Severity Level: {severity}")
#     print(model.metrics_names)
# print(model.summary())  # Check model architecture
from sklearn.metrics import silhouette_score



