import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the dataset path
dataset_path = r"C:\Users\rucha\OneDrive\Desktop\Oily-Dry-Skin-Types"

# Set AUTOTUNE for optimization
autotune = tf.data.AUTOTUNE

# Load dataset with corrected paths
train_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(dataset_path, "train"), image_size=(224, 224), batch_size=32, shuffle=True
)
valid_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(dataset_path, "valid"), image_size=(224, 224), batch_size=32, shuffle=True
)
test_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(dataset_path, "test"), image_size=(224, 224), batch_size=32, shuffle=True
)

# Get class names (should be ["Oily", "Normal", "Dry"])
class_labels = train_dataset.class_names
print("Class Labels:", class_labels)

# Normalize images
normalization_layer = layers.Rescaling(1.0 / 255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
valid_dataset = valid_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# Data augmentation
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2)
])
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

# Optimize dataset performance
train_dataset = train_dataset.shuffle(1000).cache().prefetch(autotune)
valid_dataset = valid_dataset.cache().prefetch(autotune)
test_dataset = test_dataset.cache().prefetch(autotune)

# Load EfficientNetB0 model
base_model = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights="imagenet")
base_model.trainable = False  # Freeze base model initially

# Add classification head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(len(class_labels), activation='softmax')
])

# Unfreeze last 30 layers for fine-tuning
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(train_dataset, validation_data=valid_dataset, epochs=10)

# Prediction function
def predict_skin_type(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    
    return predicted_class, predictions[0]

# Example prediction
img_path = "test_image.jpg"
predicted_class, confidence_scores = predict_skin_type(img_path, model)
print(f"Predicted Skin Type: {predicted_class}")

# Plot confidence scores
plt.bar(class_labels, confidence_scores)
plt.xlabel("Skin Types")
plt.ylabel("Confidence")
plt.title("Prediction Confidence Scores")
plt.show()

# Save the trained model
model.save("SkintelligenceMLModel.h5")
