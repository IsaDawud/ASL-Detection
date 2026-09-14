import numpy as np
import cv2
import os
from glob import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ===============================
# GET IMAGE SIZE
# ===============================
def get_image_size():
    img_path = 'gestures/0/0.jpg'
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"❌ Sample image not found: {img_path}")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return img.shape


# ===============================
# COUNT CLASSES
# ===============================
def get_num_of_classes():
    return len(glob('gestures/*'))


image_x, image_y = get_image_size()


# ===============================
# CNN MODEL
# ===============================
def cnn_model():
    num_of_classes = get_num_of_classes()

    model = Sequential([
        Conv2D(16, (2, 2), input_shape=(image_x, image_y, 1), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),

        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), padding='same'),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2), padding='same'),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_of_classes, activation='softmax')
    ])

    optimizer = optimizers.SGD(learning_rate=0.001)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    filepath = "cnn_model_keras2.h5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    return model, checkpoint


# ===============================
# TRAIN FUNCTION
# ===============================
def train():
    image_data = []
    labels = []

    print("📂 Loading images...")

    for folder in glob('gestures/*'):
        label = int(os.path.basename(folder))
        for img_path in glob(f"{folder}/*.jpg"):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (image_x, image_y))
            image_data.append(img)
            labels.append(label)

    image_data = np.array(image_data, dtype=np.float32)
    labels = np.array(labels)

    # Normalize
    image_data = image_data.reshape(-1, image_x, image_y, 1) / 255.0
    labels = to_categorical(labels)

    # Shuffle
    indices = np.arange(image_data.shape[0])
    np.random.shuffle(indices)
    image_data = image_data[indices]
    labels = labels[indices]

    # Split 80:20
    split = int(0.8 * len(image_data))
    train_images, val_images = image_data[:split], image_data[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    print("🚀 Training started...")

    model, checkpoint = cnn_model()
    model.summary()
    history = model.fit(
        train_images,
        train_labels,
        validation_data=(val_images, val_labels),
        epochs=10,
        batch_size=32,
        callbacks=[checkpoint]
    )

    # ===============================
    # ACCURACY GRAPH
    # ===============================
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"])
    plt.show()

    # ===============================
    # LOSS GRAPH
    # ===============================
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"])
    plt.show()

    # ===============================
    # CONFUSION MATRIX
    # ===============================
    y_pred = model.predict(val_images)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(val_labels, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[str(i) for i in range(get_num_of_classes())]
    )

    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

    # ===============================
    # FINAL EVALUATION
    # ===============================
    scores = model.evaluate(val_images, val_labels, verbose=0)
    print(f"\n✅ FINAL ACCURACY: {scores[1] * 100:.2f}%")
    print("💾 Model saved as cnn_model_final.h5")
    model.save('cnn_model_final.h5')



# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    train()
