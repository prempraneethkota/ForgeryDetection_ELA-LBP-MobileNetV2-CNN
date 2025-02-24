import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers

# Define Custom Layers
class ELALayer(layers.Layer):
    def __init__(self, quality=95, **kwargs):
        super().__init__(**kwargs)
        self.quality = quality

    def call(self, inputs):
        inputs = tf.image.resize(inputs, [256, 256])
        def jpeg_compress(image):
            image = tf.image.encode_jpeg(tf.cast(image, tf.uint8), quality=self.quality)
            return tf.image.decode_jpeg(image)
        compressed = tf.map_fn(jpeg_compress, inputs, dtype=tf.uint8)
        ela = tf.abs(tf.cast(inputs, tf.float32) - tf.cast(compressed, tf.float32)) / 255.0
        return ela

    def get_config(self):
        return {"quality": self.quality}

class LBPLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        grayscale = tf.image.rgb_to_grayscale(inputs) if inputs.shape[-1] == 3 else inputs
        patches = tf.image.extract_patches(images=grayscale, sizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
        center = patches[..., 4:5]
        binary = tf.where(patches >= center, 1.0, 0.0)
        kernel = tf.constant([[1, 2, 4], [128, 0, 8], [64, 32, 16]], dtype=tf.float32)
        kernel = tf.reshape(kernel, [3, 3, 1, 1])
        kernel = tf.tile(kernel, [1, 1, 9, 1])
        lbp = tf.nn.conv2d(binary, kernel, strides=[1, 1, 1, 1], padding='SAME')
        return lbp

    def get_config(self):
        return {}

# Load the trained model with custom objects
MODEL_PATH = 'forgery_detection_hybrid.h5'
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"ELALayer": ELALayer, "LBPLayer": LBPLayer}, compile=False)

# Test folder containing images
test_folder = 'test/'

def predict_image(img_path):
    """Load and predict an image using the trained model."""
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    prediction_score = model.predict(img_array, verbose=0)[0][0]
    predicted_label = "Authentic" if prediction_score < 0.5 else "Forged"
    
    return img_path, predicted_label, prediction_score

# Run predictions on test images
if __name__ == "__main__":
    for img_name in os.listdir(test_folder):
        img_path = os.path.join(test_folder, img_name)
        img_name, label, score = predict_image(img_path)
        print(f"Image: {img_name}, Prediction: {label} (Confidence: {score:.4f})")
