import cv2
import numpy as np
import os
import random

# Load the face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# Initialize the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=125)

dataset_path = "dataset"
images, labels = [], []
label_dict = {}
current_id = 0

# Define image size
IMG_SIZE = (120, 120)  # Increase from (100, 100)  

# 📌 Data Augmentation Functions
def augment_image(image):
    """Apply light augmentation without overfitting"""
    augmented_images = []

    # Flip image horizontally (mirroring)
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)

    # Small rotation (avoid large distortions)
    angle = random.choice([-3, 3])  # Small angle to keep face features
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    augmented_images.append(rotated)

    return augmented_images

# 📌 Process each person's folder
for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    if not os.path.isdir(person_path):
        continue

    label_dict[current_id] = person  # Assign an ID to the name

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        faces = face_cascade.detectMultiScale(img, 1.1, 5)
        for (x, y, w, h) in faces:
            face_roi = img[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, IMG_SIZE)  # Resize to match training size

            images.append(face_roi)
            labels.append(current_id)

            # 📌 Apply Data Augmentation
            for aug_img in augment_image(face_roi):
                images.append(aug_img)
                labels.append(current_id)

    current_id += 1  # Increment ID for next person

# Convert lists to numpy arrays
images = np.array(images, dtype=np.uint8)
labels = np.array(labels, dtype=np.int32)

# Train the recognizer
recognizer.train(images, labels)
recognizer.save('trained_model.yml')

# Save labels
with open("labels.txt", "w") as f:
    for id_, name in label_dict.items():
        f.write(f"{id_},{name}\n")

print("✅ Training complete with augmented images!")
