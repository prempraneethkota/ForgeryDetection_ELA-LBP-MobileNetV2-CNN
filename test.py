
# In[1]:


import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2


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


# In[5]:


MODEL_PATH = 'topc/forgery_detection_hybrid.h5'
model = tf.keras.models.load_model(
    MODEL_PATH, 
    custom_objects={"ELALayer": ELALayer, "LBPLayer": LBPLayer, "MobileNetV2": MobileNetV2}, 
    compile=False
)

test_folder = "topc/test/"  # Path to your test folder

for img_name in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_name)
    
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(256, 256))  # Resize to match model input
    img_array = image.img_to_array(img) / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction_score = model.predict(img_array, verbose=0)[0][0]  # Get scalar prediction
    predicted_label = "Authentic" if prediction_score < 0.5 else "Forged"

    print(f"Image: {img_name}, Prediction: {predicted_label} (Confidence: {prediction_score:.4f})")


# In[ ]:
