#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import layers, applications, models
import cv2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import numpy as np
import os
from tensorflow.keras.preprocessing import image


# In[2]:


# Custom Layer for Error Level Analysis (ELA)
class ELALayer(layers.Layer):
    def __init__(self, quality=95, **kwargs):
        super().__init__(**kwargs)
        self.quality = quality

    def call(self, inputs):
        # Ensure inputs are in the range [0, 255] and of type uint8
        if inputs.dtype != tf.uint8:
            inputs = tf.clip_by_value(inputs * 255., 0, 255)
            inputs = tf.cast(inputs, tf.uint8)

        # Resize images to a smaller dimension to avoid OpenCV limitations
        inputs = tf.image.resize(inputs, [256, 256])

        # Simulate JPEG compression and decompression using TensorFlow operations
        def jpeg_compress(image):
            # Ensure the image is in uint8 format before encoding
            if image.dtype != tf.uint8:
                image = tf.cast(image, tf.uint8)

            # Convert to JPEG (simulate compression)
            image = tf.image.encode_jpeg(image, quality=self.quality)
            # Decode back to image
            return tf.image.decode_jpeg(image)

        # Apply JPEG compression and compute ELA
        compressed = tf.map_fn(jpeg_compress, inputs, dtype=tf.uint8)
        ela = tf.abs(tf.cast(inputs, tf.float32) - tf.cast(compressed, tf.float32))
        ela = ela / 255.0  # Normalize to [0, 1]

        return ela

    def get_config(self):
        config = super().get_config()
        config.update({
            "quality": self.quality,
        })
        return config

# Custom Layer for Local Binary Patterns (LBP)
class LBPLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        if inputs.shape[-1] == 3:
            grayscale = tf.image.rgb_to_grayscale(inputs)
        else:
            grayscale = inputs

        # Extract patches around each pixel
        patches = tf.image.extract_patches(
            images=grayscale,
            sizes=[1, 3, 3, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='SAME'
        )

        # Get the center pixel value
        center = patches[..., 4:5]

        # Compare each pixel in the patch with the center pixel
        binary = tf.where(patches >= center, 1.0, 0.0)

        # Define the LBP kernel weights
        kernel = tf.constant([
            [1, 2, 4],
            [128, 0, 8],
            [64, 32, 16]
        ], dtype=tf.float32)

        # Reshape the kernel to match the patch dimensions
        kernel = tf.reshape(kernel, [3, 3, 1, 1])  # Shape: (3, 3, 1, 1)
        kernel = tf.tile(kernel, [1, 1, 9, 1])     # Shape: (3, 3, 9, 1)

        # Apply the LBP kernel to each patch
        lbp = tf.nn.conv2d(binary, kernel, strides=[1, 1, 1, 1], padding='SAME')

        return lbp

    def get_config(self):
        config = super().get_config()
        return config


# In[3]:


# Build the Hybrid Model
tf.get_logger().setLevel('ERROR')
def build_hybrid_model(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1./255)(inputs)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.1)(x)

    # Branch 1: MobileNetV2 (unchanged)
    mobilenet = applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        alpha=0.35 # To reduce model complexity
    )
    for layer in mobilenet.layers[:10]:
        layer.trainable = False
    mobilenet.trainable = False
    branch1 = mobilenet(x)
    branch1 = layers.GlobalAveragePooling2D()(branch1)
    branch1 = layers.Dropout(0.4)(branch1) # Dropouts to overcome overfitting

    # Branch 2: ELA + LBP + CNN
    ela = ELALayer(quality=95)(inputs)

    # CNN Layers for ELA output
    ela_cnn = layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer = tf.keras.initializers.HeNormal())(ela)  # Added CNN
    ela_cnn = layers.MaxPooling2D((2, 2))(ela_cnn)
    ela_cnn = layers.Flatten()(ela_cnn)  # Flatten before combining with LBP

    lbp = LBPLayer()(ela)
    lbp = layers.GlobalAveragePooling2D()(lbp)

    branch2 = layers.Concatenate()([ela_cnn, lbp])  # Combine CNN output with LBP

    branch2 = layers.Dense(128, activation='relu', kernel_initializer = tf.keras.initializers.HeNormal(), kernel_regularizer=l2(0.00001))(branch2)
    branch2 = layers.Dropout(0.5)(branch2)

    # Final Concatenation (unchanged)
    combined = layers.Concatenate()([branch1, branch2])
    combined = layers.Dropout(0.5)(combined)

    outputs = layers.Dense(1, activation='sigmoid', 
                       kernel_initializer=tf.keras.initializers.GlorotNormal())(combined)

    model = models.Model(inputs, outputs)
    return model


# In[4]:


# Define Learning Rate Decay (Prevents Overfitting)
initial_lr = 1e-3

# Compile the Model with Learning Rate Decay
model = build_hybrid_model()
model.summary()

# Dataset Preparation with Augmentation and Normalization
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,  # Normalize images
    rotation_range=20,  
    width_shift_range=0.2,  
    height_shift_range=0.2,  
    horizontal_flip=True,  
    validation_split=0.2  # 20% for validation
)

# Load Training Data
train_generator = train_datagen.flow_from_directory(
    'data/',
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Load Validation Data
val_generator = train_datagen.flow_from_directory(
    'data/',
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Check Class Distribution
print("Class Distribution in Training Set:", np.bincount(train_generator.classes))
print("Class Distribution in Validation Set:", np.bincount(val_generator.classes))

# Learning Rate Decay Steps Dynamically
steps_per_epoch = len(train_generator)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=steps_per_epoch * 40,
    decay_rate=0.999,
    staircase=True
)

# Compile the Model with Adjusted Learning Rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# In[5]:


# Define EarlyStopping Callback (Prevents Overfitting)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1,
    restore_best_weights=True
)

# Train the Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[early_stopping]  # Early stopping applied
)


# In[6]:


# Export
model.save('forgery_detection_hybrid.h5')


# In[ ]:




