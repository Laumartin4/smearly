import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0

from smearly.ml_logic.preprocessing import create_image_dataset

def initialize_cnn_model(input_shape : tuple) -> Model:
    """Initialize Neural Network model
    Args: input_shape (tuple)
    Returns: Model
    """

    model = Sequential()

    model.add(layers.Input(input_shape))

    model.add(layers.Conv2D(filters = 32, kernel_size = (3,3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="same") )


    model.add(layers.Conv2D(filters = 32, kernel_size = (3,3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="same") )


    model.add(layers.Conv2D(filters = 64, kernel_size = (3,3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="same") )

    model.add(layers.Conv2D(filters = 128, kernel_size = (3,3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="same") )

    model.add(layers.Flatten())

    model.add(layers.Dense(64, activation="relu"))

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(3, activation="softmax"))
    
    print("✅ CNN Model initialized")
    
    return model

def initialize_enb0_model(input_shape : tuple) -> Model:
    """Initialize EfficientNetB0 model

    /!\ BEWARE the input pixel values must not be normalized in this model (keep [0-255])

    Args: input_shape (tuple)
    Returns: Model
    """

    base_model = EfficientNetB0(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = False
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    prediction = layers.Dense(3, activation='softmax')(x)

    ENB0_model = Model(inputs=base_model.input, outputs=prediction)

    print("✅ EfficientNetB0 model initialized")
    return ENB0_model

def initialize_enb0_model_layers(input_shape: tuple) -> Model:
    """Initialize EfficientNetB0 model

    /!\ BEWARE the input pixel values must not be normalized in this model (keep [0-255])

    Args: input_shape (tuple)
    Returns: Model
    """

    base_model = EfficientNetB0(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = False
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    
    # Add three additional layers before the last layer
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Optional: Add dropout for regularization
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    
    # Output layer
    prediction = layers.Dense(3, activation='softmax')(x)

    ENB0_model = Model(inputs=base_model.input, outputs=prediction)

    print(":white_check_mark: EfficientNetB0 model initialized")
    return ENB0_model

def compile_model(model : Model, learning_rate = 0.001) -> Model:
    """Compile
        optimizer : str, loss : str, metrics : list
        
    """
    adam = optimizers.Adam(learning_rate = learning_rate)
    model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['f1_score', 'accuracy', 'precision', 'recall'])
    
    print("✅ Model compiled")
    return model

def train_model(
        model: Model,
        train_data,  # Can be a TensorFlow dataset or a tuple (X_train, y_train)
        validation_data=None,  # Can be a TensorFlow dataset or a tuple (X_val, y_val)
        validation_split=0.3,
        batch_size=256,
        epochs=10,
        fine_tuning=False):
    """Train model with support for TensorFlow datasets and NumPy arrays."""

    if fine_tuning:
        MODEL = f"{model}.h5"

        modelCheckpoint = callbacks.ModelCheckpoint(MODEL,
                                                    monitor="val_loss",
                                                    verbose=0,
                                                    save_best_only=True)

        LRreducer = callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                factor=0.1,
                                                patience=3,
                                                verbose=1,
                                                min_lr=0)

        EarlyStopper = callbacks.EarlyStopping(monitor='val_loss',
                                                patience=10,
                                                verbose=0,
                                                restore_best_weights=True)

        callbacks_ft = [modelCheckpoint, LRreducer, EarlyStopper]
    else:
        callbacks_ft = None

    if isinstance(train_data, tf.data.Dataset):
        # Training with TensorFlow dataset

        history = model.fit(train_data, epochs=epochs, callbacks=callbacks_ft, validation_data=validation_data)
        print(f"✅ Model trained on TensorFlow dataset with last global F1 score : {round(np.mean(history.history['f1_score'][-1]), 2)}")

    elif isinstance(train_data, tuple) and len(train_data) == 2 and isinstance(train_data[0], (np.ndarray, tf.data.Dataset, tf.Tensor)) and isinstance(train_data[1], (np.ndarray, tf.data.Dataset, tf.Tensor)):
        # Training with NumPy arrays
        X_train, y_train = train_data
        X_train = X_train / 255.0 
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                            validation_split=validation_split, callbacks=callbacks_ft)
        print(f"✅ Model trained on {len(X_train)} images with last global F1 score : {round(np.mean(history.history['f1_score'][-1]), 2)}")
    else:
        raise ValueError("train_data must be a TensorFlow dataset or a tuple (X_train, y_train) of NumPy arrays.")

    return model, history

def evaluate_model(model : Model, X_test : np.array, y_test : np.array) -> dict:
    """Evaluate model"""
    evaluation = model.evaluate(X_test, y_test)
    print(f"✅ Model evaluated on {len(X_test)} images with global F1 score : {round(np.mean(evaluation[1]),2)}")
    return evaluation

