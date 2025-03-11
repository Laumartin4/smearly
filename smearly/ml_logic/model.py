import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0


def initialize_cnn_model(input_shape : tuple) -> Model:
    """Initialize Neural Network model
    Args: input_shape (tuple)
    Returns: Model
    """

    model = Sequential()

    model.add(layers.Input((224, 224, 3)))
    model.add(layers.Rescaling(1./255))

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
    Args: input_shape (tuple)
    Returns: Model
    """

    base_model = EfficientNetB0(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = False
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    prediction = layers.Dense(3, activation='softmax')(x)

    ENB0_model = Model(inputs=base_model.input, outputs=prediction)

    print("✅ EfficientNetB0 model initialized")
    return ENB0_model


def compile_model(model : Model, learning_rate = 0.001) -> Model:
    """Compile
        optimizer : str, loss : str, metrics : list
        
    """
    adam = optimizers.Adam(learning_rate = learning_rate)
    model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['f1_score'])
    
    print("✅ Model compiled")
    return model


def train_cnn_model(
        model : Model, 
        X_train : np.array, y_train : np.array, 
        validation_split = 0.3,
        batch_size = 256, 
        epochs = 10,
        fine_tuning = False) -> Tuple[Model, dict]:
    """Train model"""
    
    if fine_tuning == True :
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
        
        callbacks = [modelCheckpoint, LRreducer, EarlyStopper]
    else :
        callbacks = None
    
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks= callbacks)
    print(f"✅ Model trained on {len(X_train)} images with max F1 score : {round(np.max(history.history['f1_score']), 2)}")
    return model, history

    
def evaluate_model(model : Model, X_test : np.array, y_test : np.array) -> dict:
    """Evaluate model"""
    evaluation = model.evaluate(X_test, y_test)
    print(f"✅ Model evaluated on {len(X_test)} images with F1 score : {round(evaluation[1], 2)}")
    return evaluation

######## A REVOIR DEMAIN ########

def predict(model : Model, X : np.array) -> np.array:
    """Predict"""
    predictions = model.predict(X)
    print("✅ Predictions made")
    return predictions


def save_model(model : Model, model_name : str) -> None:    
    model.save(model_name)
    print(f"✅ Model saved as {model_name}")
    
    
def load_model(model_name : str) -> Model:
    model = tf.keras.models.load_model(model_name)
    print(f"✅ Model loaded from {model_name}")
    return model
