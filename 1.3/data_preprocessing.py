
"""
data_preprocessing.py
CIFAR-10 Data Loading & Preprocessing Utilities
"""

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter

def load_cifar10(batch_size=64):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    X = np.concatenate((x_train, x_test))
    Y = np.concatenate((y_train, y_test))

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, Y, test_size=0.30, stratify=Y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    X_train, X_val, X_test = [arr.astype("float32") / 255.0 for arr in (X_train, X_val, X_test)]

    print("Class counts (train):", Counter(y_train.flatten()))

    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
    )
    val_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()

    train_gen = train_datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_gen   = val_datagen.flow(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_gen  = test_datagen.flow(X_test, y_test, batch_size=batch_size, shuffle=False)

    return train_gen, val_gen, test_gen, (X_train, X_val, X_test, y_train, y_val, y_test)

if __name__ == "__main__":
    tg, vg, tg2, _ = load_cifar10()
    print("✅ Data generators ready.")
