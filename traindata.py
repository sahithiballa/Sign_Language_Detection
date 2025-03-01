import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
import cv2

# Set the path to the collected data
data_folder = "D:/SAHITHI BALLA/projects/SLD_6thsem/Data"

# Set the labels for each class
labels = ["A","B","C","D","E","F","G","GoodLuck","H","Hello","I","K","L","No","Thankyou","V"]

# Create a list to store the images and labels
images = []
labels_list = []

# Loop through each class folder
for label in labels:
    folder_path = os.path.join(data_folder, label)
    for file in os.listdir(folder_path):
        # Read the image
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))  # Resize the image to 300x300
        images.append(img)
        labels_list.append(labels.index(label))  # Append the label index

# Convert the lists to numpy arrays
images = np.array(images)
#labels = np.array(labels_list)

#labels_list = [labels.index(label) for label in labels_list]
labels_onehot = tf.keras.utils.to_categorical(labels_list, num_classes=len(labels))
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels_onehot, test_size=0.2, random_state=42)

# Normalize the images
X_train = X_train / 255.0
X_test = X_test / 255.0

# Create a CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(len(labels), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50,batch_size=16, 
                    validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.2f}')

# Save the model
model.save('hand_gesture_model.h5')