# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 09:19:58 2025

@author: Student
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

def main_modal(train_dir, test_dir,val_dir): 

    """ Main function to run the entire training pipeline. """
     
    # Load data
    train_data, val_data, test_data = load_data(train_dir, test_dir, val_dir)

    # Build and train model
    model = build_model()
    history = train_model(model, train_data, val_data, epochs=5)# גם פה לשנות מספר חזרות

    # Evaluate model
  #  evaluate_model(model, test_dir)

    # Plot accuracy & loss graph
    plot_training_history(history)

    # Save model
    save_model(model)

def load_data(train_dir, test_dir,val_dir):
    # Load and preprocess data using ImageDataGenerator.
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_data = datagen.flow_from_directory(train_dir, 
                                             target_size=(224, 224),  
                                             batch_size=32, 
                                             class_mode='categorical',  
                                             subset='training')

    val_data = datagen.flow_from_directory(val_dir, 
                                           target_size=(224, 224), 
                                           batch_size=32, 
                                           class_mode='categorical', 
                                           subset='validation')

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_data = test_datagen.flow_from_directory(test_dir, 
                                                 target_size=(224, 224), 
                                                 batch_size=32, 
                                                 class_mode='categorical')
    
    return train_data, val_data, test_data

def build_model():
    # Create and compile a CNN model for classification. 
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),

        Conv2D(64, (3,3), activation='relu'),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')  
    ])

    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy',  
                  metrics=['accuracy'])

    return model

def train_model(model, train_data, val_data, epochs=5):# פה לשנות את מספר החזרות
    """ Train the model and return training history. """
    history = model.fit(train_data, 
                        validation_data=val_data, 
                        epochs=epochs)

    return history

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""
def evaluate_model(model, test_dir):

  #  Evaluates the trained model on the test dataset.
    
   # Parameters:
  #  - model: Trained Keras model
   # - test_dir: Path to the test dataset directory
    
  #  Returns:
   # - test_loss: Loss on the test set
  #  - test_acc: Accuracy on the test set
    

    # Ensure test data is loaded correctly
    test_datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),  # Adjust according to model input size
        batch_size=32,
        class_mode="categorical",  # Change if using binary classification
        shuffle=False  # No shuffling for evaluation
    )

    # Evaluate the model on the test data
    test_loss, test_acc = model.evaluate(test_generator)

    print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    return test_loss, test_acc

"""
def plot_training_history(history):
    """ Plot accuracy & loss graphs after training. """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)

    # Plot Accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training & Validation Accuracy')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training & Validation Loss')

    plt.show()

def save_model(model, filename="brain_tumor_model.h5"):
    """ Save the trained model to a file. """
    model.save(filename)
    print(f"Model saved as {filename}")